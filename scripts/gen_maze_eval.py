import argparse
import csv
from pathlib import Path
import collections
import pickle
import random
from typing import Dict, List, Tuple, Any

import numpy as np

from MazeGen import make_maze

def shortest_path_with_parents(maze: np.ndarray):
    '''
        Return (path, length) of the shortest path between agent (2) and goal (3).
    '''
    
    walls = maze[0]
    H, W = walls.shape
    start = tuple(np.argwhere(maze[1] == 1)[0])
    goal = tuple(np.argwhere(maze[2] == 1)[0])

    q: collections.deque = collections.deque([start])
    parent = {start: None}
    moves = [(-1,0),(1,0),(0,-1),(0,1)]
    while q:
        r,c = q.popleft()
        if (r,c) == goal:
            path = []
            node = (r,c)
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path, len(path)-1
        for dr,dc in moves:
            nr, nc = r+dr, c+dc
            if 0<=nr<H and 0<=nc<W and walls[nr,nc]==0 and (nr,nc) not in parent:
                parent[(nr,nc)] = (r,c)
                q.append((nr,nc))
    return [], -1

def decision_points(path: List[Tuple[int,int]], walls: np.ndarray) -> int:
    moves = [(-1,0),(1,0),(0,-1),(0,1)]
    H = walls.shape[0]
    cnt = 0
    for r,c in path[1:-1]:
        n=0
        for dr,dc in moves:
            nr,nc=r+dr,c+dc
            if 0<=nr<H and 0<=nc<H and walls[nr,nc]==0:
                n+=1
        if n>2:
            cnt+=1
    return cnt

def difficulty_bucket(maze: np.ndarray):
    walls = maze[0]
    g = walls.shape[0]
    path, plen = shortest_path_with_parents(maze)
    if plen == -1:
        return "unsolvable", plen, 0, float("inf")
    junc = decision_points(path, walls)
    score = plen + 5*junc
    if score <= 1.5*g:
        bucket="easy"
    elif score <= 2.5*g:
        bucket="normal"
    else:
        bucket="hard"
    return bucket, plen, junc, score

def relocate_goal_interior(maze, rng) :
    '''
    Move the goal (channel 2) from its current position on the boundary to a random floor tile, and turn the old goal cell into a wall.
    '''

    # old goal
    old_row, old_col = np.argwhere(maze[2] == 1)[0]

    # new goal
    H = maze.shape[1]
    free = np.argwhere((maze[0] == 0) & (maze[2] == 0))
    interior = [pos for pos in free if 0 < pos[0] < H-1 and 0 < pos[1] < H-1]
    new_row, new_col = interior[rng.integers(len(interior))]

    # swap and add wall where exit was
    maze[2, ...] = 0
    maze[2, new_row, new_col] = 1
    maze[0, old_row, old_col] = 1

    return maze

def generate_dataset(output_dir="eval_mazes", sizes=(7,11,15), per_category=100, seed=123):
    rng = np.random.default_rng(seed)
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    metadata = []

    difficulties = ["easy","normal","hard"]
    exits = ["side","interior"]

    for size in sizes:
        target_counts: Dict[Tuple[str,str],int] = {(d,e):0 for d in difficulties for e in exits}
        buckets: Dict[Tuple[str,str],List[np.ndarray]] = {(d,e):[] for d in difficulties for e in exits}

        attempts = 0
        while min(target_counts.values()) < per_category and attempts < 1_500_000:
            attempts += 1
            maze = make_maze(size)
            diff, plen, junc, score = difficulty_bucket(maze)
            if diff=="unsolvable":
                continue

            # decide exit_type to generate / prioritise category hasn't met quota yet
            if target_counts[(diff,"side")] < per_category and target_counts[(diff,"interior")] < per_category:
                exit_type = rng.choice(exits)
            elif target_counts[(diff,"side")] < per_category:
                exit_type = "side"
            elif target_counts[(diff,"interior")] < per_category:
                exit_type = "interior"
            else:
                # bucket full
                continue  

            if exit_type == "interior":
                maze = maze.copy()
                maze = relocate_goal_interior(maze, rng)
                # The move can change difficulty = re-bucket
                diff, plen, junc, score = difficulty_bucket(maze)

                # If that bucket is already full, skip this maze
                if target_counts[(diff, "interior")] >= per_category:
                    continue

            idx = target_counts[(diff,exit_type)] + 1
            buckets[(diff,exit_type)].append(maze)
            target_counts[(diff,exit_type)] += 1

            metadata.append([
                f"{size}_{diff}_{exit_type}_{idx:03d}",
                size,
                diff,
                plen,
                junc,
                score,
                exit_type,
            ])

        # Save pickles
        for (diff,exit_type), mazes in buckets.items():
            fname = Path(output_dir)/f"{size}_{diff}_{exit_type}.pkl"
            with fname.open("wb") as f:
                pickle.dump(mazes, f)
            print(f"Saved {len(mazes):3d} → {fname}")

    # Save metadata CSV
    csv_path = Path(output_dir)/"metadata.csv"
    with csv_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id","size","difficulty","path_len","junctions","score","exit_type"])
        writer.writerows(metadata)
    print(f"Metadata saved to {csv_path} (rows={len(metadata)})")

def _main():
    p = argparse.ArgumentParser(description="Generate eval set mazes (easy,normal,hard) / (side , interior)")
    p.add_argument("--output-dir", default="eval_mazes")
    p.add_argument("--per-category", type=int, default=100)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    generate_dataset(args.output_dir, (7,11,15), args.per_category, args.seed)

if __name__ == "__main__":
    _main()
