import torch
import torch.nn as nn
import wandb

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from stable_baselines3.common.vec_env import SubprocVecEnv,VecMonitor
import gymnasium as gym
from MazeEnv import MazeEnv 

import argparse
from einops import rearrange

import math

def mdl_penalty(model, delta: float = 1e-3):
    '''
    Soft MDL cost =  sum_i log2(1 + |w_i|/sigma)
    Using log1p keeps it numerically stable near zero, and /ln(2) converts natural log to log2.
    '''
    bits = 0.0
    ln2 = math.log(2.0)
    for p in model.parameters():
        bits = bits + torch.log1p(p.abs() / delta).sum() / ln2
    return bits

class PPO_MDL(PPO):
    def __init__(self, *args, mdl_coeff=0., mdl_delta=1e-3, **kwargs):
        self.mdl_coeff = mdl_coeff
        self.mdl_delta = mdl_delta
        super().__init__(*args, **kwargs)

    def train(self):
        # Regular PPO epoch
        super().train()
        if self.mdl_coeff == 0.0:
            return

        mdl_bits = mdl_penalty(self.policy, self.mdl_delta)
        mdl_loss = self.mdl_coeff * mdl_bits
        self.policy.optimizer.zero_grad(set_to_none=True)
        mdl_loss.backward()
        self.policy.optimizer.step()

        self.logger.record("train/mdl_bits", mdl_bits.detach().item())

class RecurrentPPO_MDL(RecurrentPPO):
    def __init__(self, *args, mdl_coeff=0., mdl_delta=1e-3, **kwargs):
        self.mdl_coeff = mdl_coeff
        self.mdl_delta = mdl_delta
        super().__init__(*args, **kwargs)

    def train(self):
        super().train()
        
        mdl_bits = mdl_penalty(self.policy, self.mdl_delta)
        mdl_loss = self.mdl_coeff * mdl_bits
        self.logger.record("train/mdl_bits", mdl_bits.detach().item())

        if self.mdl_coeff == 0.0:
            return
        
        self.policy.optimizer.zero_grad(set_to_none=True)
        mdl_loss.backward()
        self.policy.optimizer.step()
        


class AdamW_L1(torch.optim.AdamW):

    def __init__(self, params, l1_coeff=0.0, **kwargs):
        super().__init__(params, **kwargs)
        self.l1_coeff = l1_coeff

    @torch.no_grad()
    def step(self, closure=None):
        if self.l1_coeff != 0.0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.add_(self.l1_coeff * p.data.sign())

        return super().step(closure)

class CustomCNN(BaseFeaturesExtractor):
    '''
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    '''

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class MiniViT(BaseFeaturesExtractor):
    '''
    A small ViT:  (C, H, W) -> features_dim
    '''
    def __init__(self, observation_space, features_dim=256,
                 emb_dim=128, depth=4, n_heads=8, mlp_ratio=2.):
        super().__init__(observation_space, features_dim)
        C, H, W = observation_space.shape
        self.num_tokens = H * W

        # 1×1 conv = linear patch embed
        self.patch_embed = nn.Conv2d(C, emb_dim, kernel_size=1, stride=1)
        self.pos = nn.Parameter(torch.zeros(1, self.num_tokens, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=int(emb_dim * mlp_ratio),
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Flatten(),
            nn.Linear(emb_dim * self.num_tokens, features_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: B,C,H,W  ->  B,HW,emb
        x = self.patch_embed(x)                       # B,emb,H,W
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.pos
        x = self.transformer(x)                       # B,HW,emb
        return self.head(x)

parser = argparse.ArgumentParser()

parser.add_argument("--maze_size", type=int, default=11)
parser.add_argument("--n_prev_states", type=int, default=4)
parser.add_argument("--partial_obs", action="store_true")
parser.add_argument("--use_vit", action="store_true")
parser.add_argument("--lstm_hidden", type=int, default=128)
parser.add_argument("--max_ep_len", type=int, default=100)
parser.add_argument("--use_l1", action="store_true")
parser.add_argument("--use_l2", action="store_true")
parser.add_argument("--use_recurrent", action="store_true")
parser.add_argument("--use_mdl", action="store_true")
parser.add_argument("--mdl_coeff", type=float, default=1e-4)
parser.add_argument("--mdl_delta", type=float, default=1e-3)
parser.add_argument("--load_model", action="store_true")

args = parser.parse_args()

MAZE_SIZE = args.maze_size
N_PREV_STATES = args.n_prev_states
PARTIAL_OBS = args.partial_obs
USE_VIT = args.use_vit
LSTM_HIDDEN = args.lstm_hidden
USE_L1 = args.use_l1
USE_L2 = args.use_l2
MAX_EP_LEN = args.max_ep_len
USE_RECURRENT = args.use_recurrent
USE_MDL = args.use_mdl
LOAD_MODEL = args.load_model

loaded_model_path = "/teamspace/studios/this_studio/saved_models_2/PPO_7x7_1p5M_cnn_128_n4_partial_recurrent_MDL1e-06"
if LOAD_MODEL : 
    if 'n4' in loaded_model_path :
        N_PREV_STATES = 4
    else : N_PREV_STATES = 1

    if 'partial' in loaded_model_path : 
        PARTIAL_OBS = True
    else : 
        PARTIAL_OBS = False

    if 'recurrent' in loaded_model_path : 
        USE_RECURRENT = True
    else :
        USE_RECURRENT = False

run_name = f"PPO_{MAZE_SIZE}x{MAZE_SIZE}{'_curric_from7x7' if LOAD_MODEL else ''}_1p5M{'_vit' if USE_VIT else '_cnn'}{'_L1' if USE_L1 else ''}{'_L2' if USE_L2 else ''}_{LSTM_HIDDEN}_n{N_PREV_STATES}{'_partial' if PARTIAL_OBS else '_full'}{'_recurrent' if USE_RECURRENT else '_basic'}{'_MDL'+str(args.mdl_coeff) if USE_MDL else ''}"

policy_kwargs = dict(
    features_extractor_class=MiniViT if USE_VIT else CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=[dict(pi=[128, 128], vf=[128, 128])],
    optimizer_class=AdamW_L1,
    optimizer_kwargs=dict(
        weight_decay=1e-4 if USE_L2 else 0,
        l1_coeff=1e-6 if USE_L1 else 0,
        eps=1e-5,
    ),
)

if USE_RECURRENT:
    policy_kwargs.update(
        lstm_hidden_size=LSTM_HIDDEN,
        n_lstm_layers=1,
    )
        
def make_env():
    def _init():
        return MazeEnv(n_mazes=1000, maze_size=MAZE_SIZE, n_previous_states=N_PREV_STATES, partial_obs=PARTIAL_OBS, max_ep_len=MAX_EP_LEN)
    return _init

def make_eval_env():
    def _init():
        return MazeEnv(n_mazes=1000, maze_size=MAZE_SIZE, n_previous_states=N_PREV_STATES, partial_obs=PARTIAL_OBS, max_ep_len=MAX_EP_LEN)
    return _init

if __name__ == "__main__":

    NUM_ENVS = 12
    NUM_EVAL_ENVS = 2
    EVAL_FREQ = 10_000

    env = VecMonitor(SubprocVecEnv([make_env() for _ in range(NUM_ENVS)]))
    eval_env = VecMonitor(SubprocVecEnv([make_env() for _ in range(NUM_EVAL_ENVS)]))

    # Initialize wandb
    run = wandb.init(
        project="ppo-maze",
        name=run_name,
        monitor_gym=True,
        save_code=True,
        sync_tensorboard=True,
    )

    tb_dir = f"runs/{run.id}"

    algo_cls = RecurrentPPO_MDL if USE_RECURRENT else PPO_MDL

    if LOAD_MODEL : 
        model = algo_cls.load(loaded_model_path)
        model.set_env(env)
     
        model.policy.optimizer = AdamW_L1(
            model.policy.parameters(),
            weight_decay=1e-4 if USE_L2 else 0,
            l1_coeff=1e-6 if USE_L1 else 0,
            eps=1e-5,
        )

        model.mdl_coeff=args.mdl_coeff if args.use_mdl else 0.0
        model.mdl_delta=args.mdl_delta

    else : 
        model = algo_cls(
                "CnnLstmPolicy" if USE_RECURRENT else "CnnPolicy",
                env,
                policy_kwargs=policy_kwargs,
                n_steps=256,
                batch_size=256,
                ent_coef=0.01,
                max_grad_norm=1,
                verbose=1,
                tensorboard_log=tb_dir,
                mdl_coeff=args.mdl_coeff if args.use_mdl else 0.0,
                mdl_delta=args.mdl_delta,
        )

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=EVAL_FREQ // NUM_ENVS,
        n_eval_episodes=50,
        deterministic=True,
        render=False,
        best_model_save_path="./best_model",
        log_path="./eval_logs",
        verbose=1,
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        verbose=2,
    )

    callbacks = CallbackList([wandb_callback, eval_callback])
    
    model.learn(
        total_timesteps=1_500_000 if LOAD_MODEL else 1_000_000,
        callback=callbacks
    )

    model.save(run_name)

# Script calls
# python main.py --maze_size 15 --partial_obs --use_mdl --mdl_coeff 1e-6 --load_model --max_ep_len 200 && python main.py --maze_size 15 --partial_obs --use_mdl --mdl_coeff 1e-6 --use_l2 --load_model --max_ep_len 200
# python main.py --maze_size 15 --partial_obs --use_l2 --load_model --max_ep_len 200 && python main.py --maze_size 15 --partial_obs --use_mdl --mdl_coeff 1e-6 --use_l1 --load_model --max_ep_len 200
# python main.py --maze_size 15 --partial_obs --use_l1 --load_model --max_ep_len 200 && python main.py --maze_size 15 --partial_obs --use_mdl --mdl_coeff 1e-6 --use_l1 --use_l2 --load_model --max_ep_len 200
# python main.py --maze_size 15 --partial_obs --use_l1 --use_l2 --load_model --max_ep_len 200

# python main.py --maze_size 7 --n_prev_states 4 --partial_obs --lstm_hidden 128 --use_mdl --mdl_coeff 1e-6 --use_recurrent && python main.py --maze_size 7 --n_prev_states 4 --partial_obs --lstm_hidden 128 --use_mdl --mdl_coeff 1e-7 --use_recurrent
# python main.py --maze_size 7 --n_prev_states 4 --use_l2 --partial_obs --lstm_hidden 128 --use_mdl --mdl_coeff 1e-6 --use_recurrent && python main.py --maze_size 7 --n_prev_states 4 --partial_obs --lstm_hidden 128 --use_mdl --mdl_coeff 1e-7 --use_recurrent --use_l2

# python main.py --maze_size 7 --n_prev_states 4 --partial_obs --lstm_hidden 128 --use_mdl --mdl_coeff 0.0 --use_recurrent --use_l2

# python main.py --maze_size 15 --partial_obs --use_mdl --mdl_coeff 1e-6 --load_model

# MDL

# python main.py --maze_size 15 --n_prev_states 4 --partial_obs --lstm_hidden 128 --use_mdl --mdl_coeff 1e-6 --use_recurrent && python main.py --maze_size 15 --n_prev_states 4 --partial_obs --lstm_hidden 128 --use_mdl --mdl_coeff 5e-6 --use_recurrent
# python main.py --maze_size 15 --n_prev_states 4 --partial_obs --lstm_hidden 256 --use_mdl --mdl_coeff 1e-6 --use_recurrent && python main.py --maze_size 15 --n_prev_states 4 --partial_obs --lstm_hidden 256 --use_mdl --mdl_coeff 5e-6 --use_recurrent

# python main.py --maze_size 15 --n_prev_states 4 --partial_obs --lstm_hidden 128 --use_mdl --mdl_coeff 1e-6 --use_recurrent --use_l2 && python main.py --maze_size 15 --n_prev_states 4 --partial_obs --lstm_hidden 128 --use_mdl --mdl_coeff 1e-6 --use_recurrent --use_l1 
# python main.py --maze_size 15 --n_prev_states 4 --partial_obs --lstm_hidden 128 --use_mdl --mdl_coeff 1e-6 --use_recurrent --use_l2 --use_l1 && python main.py --maze_size 15 --n_prev_states 4 --partial_obs --lstm_hidden 128 --use_mdl --mdl_coeff 5e-6 --use_recurrent --use_l2

# && python main.py --maze_size 7 --n_prev_states 4 --partial_obs --lstm_hidden 128 --use_mdl --use_recurrent
# python main.py --maze_size 7 --n_prev_states 1 --partial_obs --lstm_hidden 128 --use_mdl && python main.py --maze_size 7 --n_prev_states 4 --partial_obs --lstm_hidden 128 --use_mdl  

# python main.py --maze_size 15 --n_prev_states 1 --partial_obs --lstm_hidden 128 && python main.py --maze_size 15 --n_prev_states 1
# python main.py --maze_size 15 --n_prev_states 1 --partial_obs --use_l1 && python main.py --maze_size 15 --n_prev_states 1 --use_l1
# python main.py --maze_size 15 --n_prev_states 1 --partial_obs --use_l2 && python main.py --maze_size 15 --n_prev_states 1 --use_l2 
# python main.py --maze_size 15 --n_prev_states 1 --partial_obs --use_l1 --use_l2 && python main.py --maze_size 15 --n_prev_states 1 --use_l1 --use_l2
