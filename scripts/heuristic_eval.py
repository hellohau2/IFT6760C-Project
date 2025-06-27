import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from MazeEnv import MazeEnv
from heuristics import HeuristicPolicy

def make_env(mazes: List[np.ndarray], size: int):
    '''
        Create an evaluation env (full observation).
    '''
    env = MazeEnv(
        n_mazes=1,
        maze_size=size,
        max_ep_len=200,
        n_previous_states=1,
        use_visited=True,
        partial_obs=False,
    )
    env.mazes = mazes
    return env

def evaluate(policy_name: str, buckets: Dict[Tuple[int, str, str], List[np.ndarray]]):
    print(f"\nTesting heuristic: {policy_name}")
    row = {"heuristic": policy_name}
    policy = HeuristicPolicy(policy_name)
    for (size, diff, exit_type), mazes in buckets.items():
        env = make_env(mazes, size)
        _, avg_step, success = env.test_policy(policy, recurrent=False)
        
        row[f"{size}_{diff}_{exit_type}_success"] = success
        row[f"{size}_{diff}_{exit_type}_avgstep"] = avg_step
        
        print(f"  {size:2d} {diff:6s} {exit_type:8s}: {success:5.1f}%, avg_step : {avg_step}")
    return row

def main(eval_dir: str, out_csv: str):

    buckets: Dict[Tuple[int, str, str], List[np.ndarray]] = {}
    for pkl in Path(eval_dir).glob("*_*.pkl"):
        parts = pkl.stem.split("_")
        if len(parts) != 3:
            print(f"Skipping unexpected file: {pkl.name}")
            continue
        size, diff, exit_type = int(parts[0]), parts[1], parts[2]
        with pkl.open("rb") as f:
            buckets[(size, diff, exit_type)] = pickle.load(f)

    diff_order = {"easy": 0, "normal": 1, "hard": 2}
    exit_order = {"side": 0, "interior": 1}
    sorted_keys = sorted(
        buckets.keys(), key=lambda x: (x[0], diff_order[x[1]], exit_order[x[2]])
    )

    heuristic_names = [
        "uniform",
        "random",
        "right_hand",
        "left_hand",
        "pledge",
        "tremaux",
        "greedy",
        "dfs",
        "bfs",
        "a_star",
    ]

    rows = [evaluate(name, buckets) for name in heuristic_names]
    df = pd.DataFrame(rows).set_index("heuristic")

    metrics = ["success", "avgstep"]
    df = df[
        [f"{s}_{d}_{e}_{m}"
        for (s, d, e) in sorted_keys
        for m in metrics]
    ]

    df.to_csv(out_csv)
    print(f"\nHeuristic benchmark saved to {out_csv}\n{df}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate heuristics on eval set")
    parser.add_argument("--eval-dir", default="eval_mazes")
    parser.add_argument("--out-csv", default="heuristic_results.csv")
    args = parser.parse_args()
    main(args.eval_dir, args.out_csv)
