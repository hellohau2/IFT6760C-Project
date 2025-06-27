import argparse
import os
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
import torch
import wandb

from ray.rllib.algorithms.ppo import PPO

from pathlib import Path

from MazeEnv import MazeEnv


def env_creator(env_config):
    return MazeEnv(**env_config)

register_env("MazeTransformerEnv", env_creator)

def build_trainer(args):
    config = (
        PPOConfig()
        .framework("torch")

        .environment("MazeTransformerEnv",
                     env_config=dict(
                         n_mazes=args.n_mazes,
                         maze_size=args.maze_size,
                         max_ep_len=args.max_ep_len,
                         n_previous_states=args.n_previous_states,
                         use_visited=args.use_visited,
                     ))

        .env_runners(
            num_env_runners=args.num_workers,
            rollout_fragment_length=args.rollout_fragment,
            compress_observations=True,
        )
        .learners(num_learners=max(1, torch.cuda.device_count()), num_gpus_per_learner=1)
        .training(
            gamma=0.99,
            lr=args.lr,
            grad_clip=args.grad_clip,
            train_batch_size_per_learner=args.train_batch_size,
            num_epochs=args.num_sgd_iter,
        )
        .rl_module(
            model_config=dict(
                # vision front-end
                conv_filters=[[32,[3,3],1],
                              [64,[3,3],1],
                              [128,[3,3],1]],
                conv_activation="relu",
                channel_major=True,        # (C,H,W)
                # gated-Transformer-XL
                use_attention=True,
                max_seq_len=64,
                attention_num_transformer_units=2,
                attention_dim=128,
                attention_num_heads=4,
                attention_head_dim=32,
                attention_memory_training=64,
                attention_memory_inference=64,
                attention_position_wise_mlp_dim=256,
            )
        )
    )
    # new builder name
    return config.build_algo()

def evaluate(algo, env, thinking_steps=10, deterministic=True):
    """
    Runs one episode but lets the model “think” for `thinking_steps`
    before the first real env step (same obs fed repeatedly).
    Returns the episode reward.
    """
    obs, _ = env.reset()
    state = algo.get_initial_state()

    for _ in range(thinking_steps):
        act, state, _ = algo.compute_single_action(
            obs,
            state=state,
            explore=False,
        )

    done = False
    total_reward = 0.0
    while not done:
        act, state, _ = algo.compute_single_action(
            obs,
            state=state,
            explore=not deterministic,
        )
        obs, rew, done, _, _ = env.step(int(act))
        total_reward += rew
    return total_reward

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--n_mazes", type=int, default=1000)
    p.add_argument("--maze_size", type=int, default=7)
    p.add_argument("--max_ep_len", type=int, default=100)
    p.add_argument("--n_previous_states", type=int, default=4)
    p.add_argument("--use_visited", action="store_true")

    p.add_argument("--training_iterations", type=int, default=10_000)
    p.add_argument("--checkpoint_interval", type=int, default=1_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=0.5)
    p.add_argument("--train_batch_size", type=int, default=4096)
    p.add_argument("--sgd_minibatch_size", type=int, default=512)
    p.add_argument("--num_sgd_iter", type=int, default=8)
    p.add_argument("--rollout_fragment", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=7)
    p.add_argument("--num_gpus", type=int, default=0)

    p.add_argument("--out_dir", type=str, default="checkpoints")
    return p.parse_args()

def main():

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    wandb.init(
        project="ppo-maze",
        name=f"PPO_Trans_{args.maze_size}x{args.maze_size}_{args.training_iterations}",
        monitor_gym=True,
        save_code=True,
        sync_tensorboard=True,
    )

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    
    algo = build_trainer(args)
    args.out_dir = Path(args.out_dir).expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # algo = PPO.from_checkpoint(str(args.out_dir))

    for i in range(args.training_iterations):
        result = algo.train()

        if "episode_reward_mean" in result:
            mean_r = result["episode_reward_mean"]
        else:
            mean_r = result["env_runners"]["episode_return_mean"]

        print(f"[{i+1:04d}]  meanR={mean_r:.2f}  "
            f"len={result['env_runners']['episode_len_mean']:.1f}")

        wandb.log({'avg_len': result['env_runners']['episode_len_mean'] , 'avg_reward': mean_r})

        if (i + 1) % args.checkpoint_interval == 0:
            path = algo.save(args.out_dir)
            print(f"Checkpoint saved")

    test_env = MazeEnv(
        n_mazes=1,
        maze_size=args.maze_size,
        max_ep_len=args.max_ep_len,
        n_previous_states=args.n_previous_states,
        use_visited=args.use_visited,
    )
    score = evaluate(algo, test_env, thinking_steps=10)
    print(f"Test reward with 10 thinking steps: {score:.2f}")

    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    main()
