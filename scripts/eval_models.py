import wandb

import argparse
import concurrent.futures as cf
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from MazeEnv import MazeEnv

from pathlib import Path
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

N_THINKING_STEPS = 1

def infer_training_size(name: str) -> int | None:
    '''
        Extract training maze‑size (7, 11, 15) from filename
    '''

    for s in (7, 11, 15):
        if re.search(fr"(?<!\d){s}[xX]{s}(?!\d)", name):
            return s
    return None


def make_env(mazes: List[np.ndarray], size: int, partial: bool, n_previous: int):
    '''
        Create a MazeEnv instance populated with the provided mazes.
    '''

    env = MazeEnv(
        n_mazes=len(mazes),
        maze_size=size,
        max_ep_len=200,
        n_previous_states=n_previous,
        use_visited=True,
        partial_obs=partial,
    )
    env.mazes = mazes
    return env

def eval_single(model_path_str: str, all_keys: Tuple[Tuple[int, str, str], ...], eval_dir: str) -> Dict[str, float | str]:
    '''
    Evaluate a single model across all maze buckets.
    '''

    buckets: Dict[Tuple[int, str, str], List[np.ndarray]] = {}
    for file in Path(eval_dir).glob("*_*.pkl"):
        size, diff, exit_type = file.stem.split("_")
        with file.open("rb") as f:
            buckets[(int(size), diff, exit_type)] = pickle.load(f)

    model_path = Path(model_path_str)
    model_name = model_path.stem
    is_partial = bool(re.search(r"partial", model_name, re.I))
    recurrent = bool(re.search(r"recurrent", model_name, re.I))
    n_prev = 4 if re.search(r"n4", model_name, re.I) else 1
    size_restrict = infer_training_size(model_name) if not is_partial else None

    ModelCls = RecurrentPPO if recurrent else PPO
    model = ModelCls.load(model_path)

    row = {"model": model_name + (f"_thinking{N_THINKING_STEPS}" if N_THINKING_STEPS > 1 else "")}

    for (size, diff, ext) in all_keys:
        col = f"{size}_{diff}_{ext}"
        print(f"{model_name} : {col}")
        if size_restrict is not None and size != size_restrict:
            row[col] = np.nan
            continue
        env = make_env(buckets[(size, diff, ext)], size, is_partial, n_prev)
        *_, sr = env.test_policy(model, recurrent=recurrent, thinking_steps = N_THINKING_STEPS)
        row[col] = sr
    return row

def run_evaluation(models_dir: str, eval_dir: str, out_csv: str, n_proc: int):
    '''
        Discover models & mazes, launch a process pool, aggregate results.
    '''

    # Build ordered list of bucket keys without loading the pickles
    diff_order = {"easy": 0, "normal": 1, "hard": 2}
    exit_order = {"side": 0, "interior": 1}

    all_keys_set: set[Tuple[int, str, str]] = set()
    for file in Path(eval_dir).glob("*_*.pkl"):
        size_str, diff, exit_type = file.stem.split("_")
        all_keys_set.add((int(size_str), diff, exit_type))
    all_keys: Tuple[Tuple[int, str, str], ...] = tuple(
        sorted(all_keys_set, key=lambda x: (x[0], diff_order[x[1]], exit_order[x[2]]))
    )
    column_order = [f"{s}_{d}_{e}" for (s, d, e) in all_keys]

    # Gather model paths
    model_paths = sorted(Path(models_dir).glob("*.zip"))
    if not model_paths:
        print("No .zip models found in", models_dir)
        return

    # Launch process pool
    max_workers = n_proc if n_proc > 0 else (os.cpu_count() or 1)
    print(f"Evaluating {len(model_paths)} models on {max_workers} processes…")

    rows: List[Dict[str, float | str]] = []
    with cf.ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(eval_single, str(p), all_keys, eval_dir) for p in model_paths
        ]
        for fut in cf.as_completed(futures):
            rows.append(fut.result())

    df = pd.DataFrame(rows).set_index("model")
    df = df[column_order]
    df.to_csv(out_csv)
    print(f"\nWrote {out_csv}  (rows={df.shape[0]}, cols={df.shape[1]})")

def _parse_args():
    p = argparse.ArgumentParser(description="trained models eval")
    p.add_argument("--models_dir", default="best_models" if N_THINKING_STEPS > 1 else "saved_models_2")
    p.add_argument("--eval_dir", default="eval_mazes")
    p.add_argument("--out_csv", default="evals_finals.csv" if N_THINKING_STEPS == 1 else f"evals_best_thinking_{N_THINKING_STEPS}"+".csv")
    p.add_argument("--n_proc", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    wandb.init(project="ppo-maze", name= "evals_finals" if N_THINKING_STEPS == 1 else f"evals_best_thinking_{N_THINKING_STEPS}")

    run_evaluation(args.models_dir, args.eval_dir, args.out_csv, args.n_proc)
