import matplotlib.pyplot as plt

import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from MazeEnv import MazeEnv
from heuristics import HeuristicPolicy

from sklearn.model_selection import train_test_split

MAX_EP_LEN = 200
THINKING_STEPS = 1
IOU_ONLY=True

def _run_episode(
    env: MazeEnv,
    model,
    *,
    wipe_memory: bool,
    recurrent: bool,
) -> bool:
    '''Play one episode and return **True** iff the maze is solved.'''
    obs, _ = env.reset()
    
    lstm_state = None
    first_step = True
    done = False
    steps = 0

    while not done and steps < MAX_EP_LEN:
        if wipe_memory and lstm_state is not None:
            lstm_state = tuple(np.zeros_like(s) for s in lstm_state)

        if recurrent:
            action, lstm_state = model.predict(
                obs,
                state=lstm_state,
                episode_start=np.array([first_step]),
                deterministic=True,
            )
        else:
            action, _ = model.predict(obs, deterministic=True)

        first_step = False
        obs, _, done, truncated, _ = env.step(int(action))
        done = done or truncated
        steps += 1

    return done and env.curr_dist_goal == 0

def _success_rate_on_mazes(
    model,
    mazes: List[np.ndarray],
    *,
    size: int,
    n_prev: int,
    wipe_memory: bool,
) -> float:
    '''Compute success-rate on **exactly** the provided mazes.'''
    env = MazeEnv(
        n_mazes=1,
        maze_size=size,
        n_previous_states=n_prev,
        use_visited=True,
        partial_obs=True,
        max_ep_len=MAX_EP_LEN,
    )
    env.mazes = mazes
    recurrent = isinstance(model, RecurrentPPO)

    successes = 0
    for maze in mazes:
        env.reset(maze=maze.copy())
        successes += _run_episode(env, model, wipe_memory=wipe_memory, recurrent=recurrent)

    return 100.0 * successes / len(mazes)


ACTION_TO_VEC = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def _train_linear(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    device: torch.device,
    task: str,
    epochs: int = 20,
    batch: int = 2048,
):
    if task not in {"cls", "bce"}:
        raise ValueError("task must be 'cls' or 'bce'")

    X_t = torch.from_numpy(X).float().to(device)

    if task == "cls":
        Y_t = torch.from_numpy(Y.squeeze()).long().to(device)
        out_dim = int(Y_t.max().item() + 1)
        loss_fn = nn.CrossEntropyLoss()
    else:                                   # BCE
        if Y.ndim != 2:
            raise ValueError("For BCE task Y must be 2-D.")
        Y_t = torch.from_numpy(Y).float().to(device)
        out_dim = Y.shape[1]
        loss_fn = nn.BCEWithLogitsLoss()

    ds = TensorDataset(X_t, Y_t)
    dl = DataLoader(ds, batch_size=min(batch, len(ds)), shuffle=True)

    model = nn.Linear(X.shape[1], out_dim, device=device)
    opt = optim.Adam(model.parameters(), lr=2e-2)

    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    return model


def _bfs_next_action(env: MazeEnv):
    planner = HeuristicPolicy("bfs")
    plan = planner._bfs_plan(
        tuple(env.agent_pos), tuple(env.target_pos), env.current_maze[0].astype(bool)
    )
    return plan[0] if plan else None


def _collect_plan_dataset(
    model,
    mazes: List[np.ndarray],
    *,
    size: int,
    n_prev: int,
    recurrent: bool,
    frames: int = 15_000,
):
    '''Collect (hidden-state/obs -> next-action) pairs on eval mazes only.'''
    env = MazeEnv(
        n_mazes=len(mazes),
        maze_size=size,
        n_previous_states=n_prev,
        use_visited=True,
        partial_obs=True,
        max_ep_len=MAX_EP_LEN,
    )
    env.mazes = mazes

    X_hid, X_obs, y = [], [], []
    lstm_state, first = None, True
    maze_idx = 0

    while len(y) < frames:
        obs, _ = env.reset(maze=mazes[maze_idx % len(mazes)].copy())
        maze_idx += 1
        done = False
        first = True
        lstm_state = None

        while not done and len(y) < frames:
            label = _bfs_next_action(env)
            if label is not None:
                y.append(label)
                X_obs.append(obs.ravel())
                if recurrent:
                    act, lstm_state = model.predict(
                        obs,
                        state=lstm_state,
                        episode_start=np.array([first]),
                        deterministic=True,
                    )
                    X_hid.append(lstm_state[0][0, 0, :].copy())
                else:
                    act, _ = model.predict(obs, deterministic=True)
                    X_hid.append(obs.ravel())
            else:
                act = env.action_space.sample()

            first = False
            obs, _, done, trunc, _ = env.step(int(act))
            done = done or trunc

    return np.vstack(X_hid), np.vstack(X_obs), np.array(y)


def _plan_probe_accuracy(X: np.ndarray, y: np.ndarray, device: torch.device):

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=0, shuffle=True
    )

    lin = _train_linear(X_tr, y_tr, device=device, task="cls")
    with torch.no_grad():
        pred = lin(torch.from_numpy(X_te).float().to(device)).argmax(1).cpu().numpy()
    return float((pred == y_te).mean())


def _collect_map_dataset(
    model,
    mazes: List[np.ndarray],
    *,
    size: int,
    n_prev: int,
):
    env = MazeEnv(
        n_mazes=len(mazes),
        maze_size=size,
        n_previous_states=n_prev,
        use_visited=True,
        partial_obs=True,
        max_ep_len=MAX_EP_LEN,
    )
    env.mazes = mazes

    X, Y = [], []
    recurrent = isinstance(model, RecurrentPPO)
    lstm_state, first = None, True

    for maze in mazes:
        obs, _ = env.reset(maze=maze.copy())
        done = False
        lstm_state, first = None, True

        while not done:
            act, lstm_state = model.predict(
                obs,
                state=lstm_state,
                episode_start=np.array([first]),
                deterministic=True,
            )

            hidden_dim = lstm_state[0].shape[-1]
            rand_vec   = np.random.randn(hidden_dim)

            X.append(rand_vec)
            # X.append(lstm_state[0][0, 0, :].copy())
            
            Y.append(env.current_maze[0].astype(np.float32).ravel())
            first = False
            obs, _, done, trunc, _ = env.step(int(act))
            done = done or trunc

    return np.vstack(X), np.vstack(Y)


def _map_iou(X: np.ndarray, Y: np.ndarray,
             device: torch.device,
             return_pred: bool = False):

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=0.20, random_state=0, shuffle=True
    )

    lin = _train_linear(X_tr, Y_tr, device=device, task="bce",
                        batch=1024, epochs=25)
    with torch.no_grad():
        logits = lin(torch.from_numpy(X_te)
                       .float()
                       .to(device)).cpu().numpy()
    pred = (logits > 0).astype(int)
    inter = np.logical_and(pred, Y_te).sum(1)
    union = np.logical_or(pred, Y_te).sum(1) + 1e-8
    iou_score = float(np.mean(inter / union))

    return (iou_score, pred) if return_pred else iou_score

def _load_buckets(eval_dir: str) -> Dict[Tuple[int, str, str], List[np.ndarray]]:
    buckets: Dict[Tuple[int, str, str], List[np.ndarray]] = {}
    for p in Path(eval_dir).glob("*_*.pkl"):
        size_s, diff, exit_t = p.stem.split("_")
        with p.open("rb") as f:
            buckets[(int(size_s), diff, exit_t)] = pickle.load(f)
    return buckets


def _is_recurrent(path: Path) -> bool:
    return "recurrent" in path.stem.lower()


def _load_model(path: Path, device: torch.device):
    cls = RecurrentPPO if _is_recurrent(path) else PPO
    mdl = cls.load(path, device=device)
    mdl.policy.to(device)
    return mdl

def _save_map_comparison(pred_vec: np.ndarray,
                         true_vec: np.ndarray,
                         size: int,
                         file_name: str) -> None:
    """
    Save a PNG with two panels:
        left  – ground-truth occupancy grid
        right – decoded occupancy grid (linear probe prediction)
    """

    size = 2 * size + 1

    pred_grid  = pred_vec.reshape(size, size)
    true_grid  = true_vec.reshape(size, size)

    fig, axes = plt.subplots(1, 2, figsize=(4, 2), dpi=200)
    for ax, grid, title in zip(
            axes,
            (true_grid, pred_grid),
            ("Ground truth", "Decoded (probe)")):
        ax.imshow(grid, cmap="gray_r")
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(file_name, bbox_inches="tight")
    plt.close(fig)

def evaluate_single(
    model_path: str,
    *,
    eval_dir: str,
    n_prev: int,
    device_type: str,
) -> Dict[str, float | str]:
    '''Evaluate a single model on **all** maze buckets + probes.'''
    device = torch.device(device_type)

    buckets = _load_buckets(eval_dir)
    path = Path(model_path)
    model = _load_model(path, device)
    recurrent = isinstance(model, RecurrentPPO)

    if IOU_ONLY:
        # if not isinstance(model, RecurrentPPO):
        #     raise ValueError("IoU probe requires a recurrent model")

        probe_size = max(k[0] for k in buckets.keys())
        probe_mazes = [
            m for (sz, _, _), maz in buckets.items() if sz == probe_size for m in maz
        ]

        X_map, Y_map = _collect_map_dataset(
            model, probe_mazes, size=probe_size, n_prev=n_prev
        )

        iou, pred_maps = _map_iou(X_map, Y_map, device, return_pred=True)

        out_png = f"{path.stem}_map.png"
        _save_map_comparison(pred_maps[0],
                            Y_map[0],
                            probe_size,
                            out_png)
        print(f"png done : {out_png}")

        return {"model": path.stem, "map_IoU": iou}

    # Determine if this model is restricted to a specific size (non-partial models)
    restrict_size = None
    if "partial" not in path.stem.lower():
        for s in (7, 11, 15):
            if f"{s}x{s}" in path.stem:
                restrict_size = s
                break

    # SR
    total_solved, total_solved_wipe, total_mazes = 0, 0, 0
    per_bucket_sr = {}

    for (size, diff, ext), mazes in buckets.items():
        if restrict_size is not None and size != restrict_size:
            per_bucket_sr[f"sr_{size}_{diff}_{ext}"] = np.nan
            continue

        sr_now = _success_rate_on_mazes(
            model,
            mazes,
            size=size,
            n_prev=n_prev,
            wipe_memory=False,
        )
        sr_now_wipe = _success_rate_on_mazes(
            model,
            mazes,
            size=size,
            n_prev=n_prev,
            wipe_memory=True,
        )

        per_bucket_sr[f"sr_{size}_{diff}_{ext}"] = sr_now

        total_solved += sr_now / 100 * len(mazes)
        total_solved_wipe += sr_now_wipe / 100 * len(mazes)
        total_mazes += len(mazes)

    sr_full = 100.0 * total_solved / total_mazes
    sr_wipe = 100.0 * total_solved_wipe / total_mazes

    print(f"\nTotal maze solved rate : {total_solved} / {total_mazes}\nTotal wiped maze solved rate : {total_solved_wipe} / {total_mazes}")

    # pick one representative size (15 if available, else max size present)
    candidate_sizes = sorted({k[0] for k in buckets.keys()})[::-1]
    probe_size = candidate_sizes[0]
    probe_mazes = []
    for (sz, diff, ext), mazes in buckets.items():
        if sz == probe_size:
            probe_mazes.extend(mazes)

    X_hid, X_obs, y_plan = _collect_plan_dataset(
        model,
        probe_mazes,
        size=probe_size,
        n_prev=n_prev,
        recurrent=recurrent,
    )
    acc_hid = _plan_probe_accuracy(X_hid, y_plan, device)
    acc_obs = _plan_probe_accuracy(X_obs, y_plan, device)

    iou = np.nan
    if recurrent:
        X_map, Y_map = _collect_map_dataset(
            model,
            probe_mazes,
            size=probe_size,
            n_prev=n_prev,
        )
        iou, pred_maps = _map_iou(X_map, Y_map, device, return_pred=True)

        out_png = f"{path.stem}_map.png"
        _save_map_comparison(pred_maps[0],
                            Y_map[0],
                            probe_size,
                            out_png)
        print(f"png done : {out_png}")

    return {
        "model": path.stem,
        "sr_full": sr_full,
        "sr_wipe": sr_wipe,
        "plan_probe_hid": acc_hid,
        "plan_probe_obs": acc_obs,
        "map_IoU": iou,
        **per_bucket_sr,
    }

def evaluate_models(
    models_dir: str,
    eval_dir: str,
    n_prev: int,
    device_choice: str,
    n_proc: int,
    out_csv: str,
):
    paths = list(Path(models_dir).glob("*.zip"))
    if not paths:
        print("⚠️  No models found in", models_dir)
        return

    n_proc = n_proc or min(os.cpu_count() or 1, len(paths))
    print(f"Evaluating {len(paths)} models on {n_proc} processes… (device {device_choice})")

    rows: List[Dict[str, float | str]] = []
    with ProcessPoolExecutor(max_workers=n_proc) as pool:
        futures = [
            pool.submit(
                evaluate_single,
                str(p),
                eval_dir=eval_dir,
                n_prev=n_prev,
                device_type=device_choice,
            )
            for p in paths
        ]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            rows.append(fut.result())

    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(out_csv)
    print(f"\nSaved {out_csv}  (rows={df.shape[0]}, cols={df.shape[1]})")
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Deterministic Mesa metrics on maze set")
    ap.add_argument("--models_dir", default="best_models")
    ap.add_argument("--eval_dir", default="eval_mazes", help="Directory with *_*.pkl mazes")
    ap.add_argument("--n_prev", type=int, default=4, help="n_previous_states")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--n_proc", type=int, default=0, help="#processes (0 uses CPU count)")
    ap.add_argument("--out_csv", default="mesa_metrics_fixed.csv")
    args = ap.parse_args()

    evaluate_models(
        args.models_dir,
        eval_dir=args.eval_dir,
        n_prev=args.n_prev,
        device_choice=args.device,
        n_proc=args.n_proc,
        out_csv=args.out_csv,
    )
