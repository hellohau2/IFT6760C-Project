from pathlib import Path
import argparse
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym 
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from MazeEnv import MazeEnv

MAZE_SIZE = 15

def infer_is_partial(name: str) -> bool:
    return bool(re.search(r"partial", name, re.I))


def infer_is_recurrent(name: str) -> bool:
    return bool(re.search(r"recurrent", name, re.I))


def infer_n_prev(name: str) -> int:
    return 4 if re.search(r"n4", name, re.I) else 1


def load_all_mazes(eval_dir: Path) -> list[np.ndarray]:

    all_mazes: list[np.ndarray] = []
    for pkl in sorted(eval_dir.glob(f"{MAZE_SIZE}_*_*.pkl")):
        with pkl.open("rb") as f:
            all_mazes.extend(pickle.load(f))
    if not all_mazes:
        raise FileNotFoundError(f"No {MAZE_SIZE}×{MAZE_SIZE} mazes found in", eval_dir)
    return all_mazes

def main(model_path: Path, eval_dir: Path, thinking_steps: int = 1) -> None:

    model_name = model_path.stem
    recurrent = infer_is_recurrent(model_name)
    partial = infer_is_partial(model_name)
    n_prev = infer_n_prev(model_name)

    ModelCls = RecurrentPPO if recurrent else PPO
    model = ModelCls.load(model_path)

    mazes = load_all_mazes(eval_dir)
    H = mazes[0].shape[1]
    W = mazes[0].shape[2]

    env = MazeEnv(
        n_mazes=len(mazes),
        maze_size=MAZE_SIZE,
        max_ep_len=200,
        n_previous_states=n_prev,
        use_visited=True,
        partial_obs=partial,
    )
    env.mazes = mazes

    heatmap = np.zeros((H, W), dtype=np.int32)

    for maze in mazes:
        obs, _ = env.reset(maze=maze.copy())

        lstm_states = None
        first_step = True
        done = False

        heatmap[tuple(env.agent_pos)] += 1

        while not done:

            if thinking_steps > 1:
                for _ in range(thinking_steps - 1):
                    if recurrent:
                        _, lstm_states = model.predict(
                            obs,
                            state=lstm_states,
                            episode_start=np.array([False]),
                            deterministic=True,
                        )
                    else:
                        model.predict(obs, deterministic=True)

            if recurrent:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=np.array([first_step]),
                    deterministic=True,
                )
            else:
                action, _ = model.predict(obs, deterministic=True)

            first_step = False
            obs, _, done, truncated, _ = env.step(int(action))
            done = done or truncated

            heatmap[tuple(env.agent_pos)] += 1

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap="hot", interpolation="nearest")
    plt.title(f"Agent position frequency : {MAZE_SIZE}×{MAZE_SIZE} mazes")
    plt.colorbar(label="visit count")
    plt.axis("off")
    out_path = Path(f"agent_position_heatmap_size_{MAZE_SIZE}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved heat-map to {out_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visitation heat-map.")
    parser.add_argument("--model_path", default="best_models/PPO_7x7_1p5M_cnn_L2_128_n4_partial_recurrent_MDL1e-06", type=Path)
    parser.add_argument("--eval_dir", default="eval_mazes", type=Path)
    parser.add_argument("--thinking_steps", default=1, type=int)
    parser.add_argument("--maze_size", default=15, type=int)
    args = parser.parse_args()

    MAZE_SIZE = args.maze_size

    main(args.model_path, args.eval_dir, args.thinking_steps)
