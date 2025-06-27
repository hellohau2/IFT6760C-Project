import os
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from MazeEnv import MazeEnv
from heuristics import HeuristicPolicy


def load_model(path: str):
    '''
    Load a model from disk, inferring whether it's recurrent by filename.
    '''
    if 'recurrent' in path.lower():
        return RecurrentPPO.load(path)
    else:
        return PPO.load(path)


def get_global_obs(env: MazeEnv) -> np.ndarray:
    '''
    Construct a full-observability frame for heuristics: walls, agent, target, visited.
    '''
    walls = env.current_maze[0]
    agent = env.current_maze[1]
    target = env.current_maze[2]
    visited = getattr(env, 'visited', np.zeros_like(walls))
    return np.stack([walls, agent, target, visited], axis=0)


def compute_optimality(model, env: MazeEnv, n_episodes: int = 50) -> float:
    '''
    Average ratio of agent's path length to true shortest-path length.
    '''
    ratios = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        steps = 0
        opt_len = env.shortest_path_length()
        done = False
        while not done and steps < env.max_ep_len:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(int(action))
            done = done or truncated
            steps += 1
        if opt_len > 0:
            ratios.append(steps / opt_len)
    return float(np.mean(ratios))


def compute_replanning(model, env: MazeEnv, injection_step: int = 10, n_episodes: int = 50) -> float:
    '''
    Inject a wall mid-episode and measure average ratio of agent's remaining steps to new optimal.
    '''
    ratios = []
    planner = HeuristicPolicy('bfs')
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        # advance to injection point
        for _ in range(injection_step):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(int(action))
            done = done or truncated
            if done:
                break
        if done:
            continue
        # use global state for planning
        walls = env.current_maze[0].astype(bool)
        pos = tuple(env.agent_pos)
        goal = tuple(env.target_pos)
        plan = planner._bfs_plan(pos, goal, walls)
        if not plan:
            continue
        # block the first step of that plan
        move = plan[0]
        dr, dc = planner.ACTION_TO_VEC[move]
        env.current_maze[0][pos[0]+dr, pos[1]+dc] = 1
        # let agent finish
        future_steps = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(int(action))
            future_steps += 1
            if done or truncated:
                break
        new_opt = env.shortest_path_length()
        if new_opt > 0:
            ratios.append(future_steps / new_opt)
    return float(np.mean(ratios))


def compute_action_agreements(model, env: MazeEnv, heuristics: list, n_episodes: int = 100) -> dict:
    '''
    Returns mean action-agreement rates for each heuristic in `heuristics`.
    '''
    results = {}
    for hname in heuristics:
        policy = HeuristicPolicy(hname)
        agrees = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            total, agree = 0, 0
            while not done:
                # model action based on partial obs
                action, _ = model.predict(obs, deterministic=True)
                # heuristic action on full obs
                obs_full = get_global_obs(env)
                h_action, _ = policy.predict(obs_full)
                if action == h_action:
                    agree += 1
                total += 1
                obs, _, done, truncated, _ = env.step(int(action))
                done = done or truncated
            if total > 0:
                agrees.append(agree / total)
        results[f'agree_{hname}'] = float(np.mean(agrees))
    return results


def evaluate_models(models_dir: str = 'best_models',
                    maze_size: int = 11,
                    n_prev_states: int = 4,
                    partial_obs: bool = True,
                    n_episodes: int = 50) -> pd.DataFrame:
    '''
    Evaluate all partial-observability models in `models_dir` on:
      - Suboptimality ratio
      - Replanning ratio
      - Action agreement across classical heuristics.
    Outputs results to 'mesa_metrics_results.csv'.
    '''
    heuristic_list = [
        'random', 'uniform', 'right_hand', 'left_hand',
        'pledge', 'tremaux', 'greedy', 'dfs',
        'bfs', 'dijkstra', 'a_star'
    ]

    records = []
    for fname in sorted(os.listdir(models_dir)):
        if not fname.endswith('.zip') or 'partial' not in fname.lower():
            continue
        path = os.path.join(models_dir, fname)
        model = load_model(path)
        env = MazeEnv(
            n_mazes=n_episodes,
            maze_size=maze_size,
            n_previous_states=n_prev_states,
            use_visited=True,
            partial_obs=partial_obs
        )

        row = {
            'model': fname,
            'opt_ratio': compute_optimality(model, env, n_episodes),
            'replan_ratio': compute_replanning(model, env, injection_step=10, n_episodes=n_episodes)
        }
        agreements = compute_action_agreements(model, env, heuristic_list, n_episodes)
        row.update(agreements)
        records.append(row)
        print(f"Evaluated {fname}")

    df = pd.DataFrame(records)
    df.to_csv('mesa_metrics_results.csv', index=False)
    print("Saved mesa_metrics_results.csv")
    return df


if __name__ == '__main__':
    evaluate_models()
