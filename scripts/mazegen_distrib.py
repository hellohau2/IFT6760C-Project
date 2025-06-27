import argparse
import collections
from typing import Dict, Tuple, List

import numpy as np

from MazeGen import make_maze

def shortest_path_with_parents(maze: np.ndarray):
    walls = maze[0]
    H, W = walls.shape
    start = tuple(np.argwhere(maze[1] == 1)[0])
    goal = tuple(np.argwhere(maze[2] == 1)[0])

    q: collections.deque = collections.deque([start])
    parent = {start: None}
    moves = [(-1,0),(1,0),(0,-1),(0,1)]

    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            path = []
            node = (r, c)
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path, len(path) - 1
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if (0 <= nr < H and 0 <= nc < W
                    and walls[nr, nc] == 0
                    and (nr, nc) not in parent):
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))
    # Unsolvable
    return [], -1


def decision_points(path: List[Tuple[int, int]], walls: np.ndarray) -> int:
    moves = [(-1,0),(1,0),(0,-1),(0,1)]
    cnt = 0
    H = walls.shape[0]
    for r, c in path[1:-1]:
        branches = 0
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < H and walls[nr, nc] == 0:
                branches += 1
        if branches > 2:
            cnt += 1
    return cnt


def difficulty_bucket(maze: np.ndarray):
    walls = maze[0]
    g = walls.shape[0]
    path, plen = shortest_path_with_parents(maze)
    if plen == -1:
        return "unsolvable", plen, 0, float("inf")
    junc = decision_points(path, walls)
    score = plen + 5 * junc
    if score <= 1.5 * g:
        bucket = "easy"
    elif score <= 2.5 * g:
        bucket = "normal"
    else:
        bucket = "hard"
    return bucket, plen, junc, score


def relocate_goal_interior(maze: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Move the goal from a boundary cell to a random interior floor tile."""
    old_r, old_c = np.argwhere(maze[2] == 1)[0]
    H = maze.shape[1]
    free = np.argwhere((maze[0] == 0) & (maze[2] == 0))
    interior = [pos for pos in free if 0 < pos[0] < H-1 and 0 < pos[1] < H-1]
    new_r, new_c = interior[rng.integers(len(interior))]
    maze[2, ...] = 0
    maze[2, new_r, new_c] = 1
    maze[0, old_r, old_c] = 1
    return maze

def run_for_size(size: int, n: int, rng: np.random.Generator) -> Dict[Tuple[str, str], int]:
    difficulties = ["easy", "normal", "hard"]
    exits = ["side", "interior"]
    counts: Dict[Tuple[str, str], int] = {(d, e): 0 for d in difficulties for e in exits}

    generated = 0
    while generated < n:
        maze = make_maze(size)

        # exit type (50% chance of relocating)
        exit_type = rng.choice(exits)
        if exit_type == "interior":
            maze = maze.copy()
            maze = relocate_goal_interior(maze, rng)

        diff, plen, junc, score = difficulty_bucket(maze)
        if diff == "unsolvable":
            continue

        counts[(diff, exit_type)] += 1
        generated += 1

    return counts


def print_table(size: int, counts: Dict[Tuple[str, str], int], total: int):
    print(f"\nMaze size {size}×{size}")
    header = f"{'Bucket':<20}{'Count':>8}{'Percent':>11}"
    print(header)
    print("-" * len(header))

    for diff in ["easy", "normal", "hard"]:
        for exit_type in ["side", "interior"]:
            c = counts[(diff, exit_type)]
            pct = 100.0 * c / total
            print(f"{diff}_{exit_type:<18}{c:8d}{pct:10.2f}%")

    print("-" * len(header))
    print(f"{'TOTAL':<20}{total:8d}{'100.00%':>11}")


def main(n_per_size: int = 1000, seed: int = 42, sizes=(7, 11, 15)):
    rng = np.random.default_rng(seed)

    grand_total = 0
    grand_counts: Dict[Tuple[str, str], int] = {(d, e): 0
        for d in ["easy", "normal", "hard"] for e in ["side", "interior"]}

    for size in sizes:
        counts = run_for_size(size, n_per_size, rng)
        print_table(size, counts, n_per_size)

        grand_total += n_per_size
        for k, v in counts.items():
            grand_counts[k] += v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args.n_per_size, args.seed)
