import numpy as np
import gymnasium as gym
import imageio
from io import BytesIO
from PIL import Image
from IPython.display import Image as IPImage, display
from MazeGen import make_maze

class MazeEnv(gym.Env):

    def __init__(self,n_mazes,maze_size,max_ep_len=100, n_previous_states=2, use_visited=True, partial_obs = False, crop_size=7) :
        '''
            Maze is a 4 x maze_size x maze_size, where each represents respectively : Wall, Floor, Agent position, Target position, and are integers of [0,1].
            (In practice the mazes are 4 x (maze_size * 2) + 1 x (maze_size * 2) + 1) from the wall generation process : a 7x7 -> 15x15.
        '''

        self.mazes = [make_maze(maze_size) for _ in range(n_mazes)]

        channels = 4 if use_visited else 3
        self.crop_size = crop_size
        
        if partial_obs : 
            self.observation_space = gym.spaces.Box(low=0,high=1,shape=(channels * n_previous_states,crop_size,crop_size), dtype=np.float32)
        else : 
            self.observation_space = gym.spaces.Box(low=0,high=1,shape=(channels * n_previous_states,maze_size * 2 + 1,maze_size * 2 + 1), dtype=np.float32)

        self.action_space = gym.spaces.Discrete(4)

        # Row , col (y,x)
        self.action_to_direction = {
            0 : np.array([-1,0]),
            1 : np.array([1,0]),
            2 : np.array([0,-1]),
            3 : np.array([0,1])
        }

        self.curr_ep_steps = 0
        self.max_ep_len = max_ep_len

        self.curr_maze_index = 0

        self.previous_states = []
        self.n_previous_states = n_previous_states
        self.use_visited = use_visited
        self.partial_obs = partial_obs

        self.reset()

        print(self.current_maze.shape)
    
    def reset(self, maze=None, seed=None,options=None):
        '''
        Resets the environment (sample a new maze)
        '''

        super().reset(seed=seed)
        self.curr_ep_steps = 0

        if maze is not None : 
            self.current_maze = maze
        else : 
            rand_index = np.random.randint(low=0,high=len(self.mazes))
            self.current_maze = self.mazes[rand_index].copy()

            # self.curr_maze_index += 1
            # self.current_maze = self.mazes[self.curr_maze_index % len(self.mazes)].copy()

        k = np.random.randint(4)
        self.current_maze = np.rot90(self.current_maze, k, axes=(1, 2))

        if self.use_visited : 
            first_frame = np.concatenate(
                [self.current_maze.copy(), (np.zeros_like(self.current_maze[0],dtype=np.float32))[None,...]],
                axis=0
            )

            self.previous_states = [first_frame.copy() for _ in range(self.n_previous_states)]
        else : 
            self.previous_states = [self.current_maze.copy() for _ in range(self.n_previous_states)]

        # row,col (y,x)
        self.agent_pos = np.array(np.argwhere(self.current_maze[1] == 1)[0]) 
        self.start_pos = self.agent_pos.copy() 
        
        self.target_pos = np.array(np.argwhere(self.current_maze[2] == 1)[0]) 
        
        self.visited = np.zeros_like(self.current_maze[0],dtype=np.float32)
        self.visited[tuple(self.agent_pos)] = 1

        self.curr_dist_goal = self.dist_from_goal()

        info = {}
        return self.get_obs(), info

    def render(self, dim_factor = 0.3):
        '''
        Convert maze channels into RGB image.
        
        Color mapping :
            - Agent (blue): [0, 0, 255]
            - Target (green): [0, 255, 0]
            - Floor (white): [255, 255, 255]
            - Wall (black): [0, 0, 0]
            
        Output:
            rgb_maze: [H, W, 3] array
        '''
        
        maze_size = self.current_maze.shape[1]
        rgb_maze = np.zeros((maze_size, maze_size, 3), dtype=np.uint8)
        
        walls, agent, target = self.current_maze
        # walls, floors, agent, target = self.current_maze
        
        rgb_maze[walls == 1] = [0, 0, 0]
        rgb_maze[walls == 0] = [255, 255, 255]
        rgb_maze[target == 1] = [0, 255, 0]
        rgb_maze[agent == 1] = [0, 0, 255]

        if self.partial_obs:
            half = self.crop_size // 2
            row, col = self.agent_pos

            # Mask for what is visible
            visible_mask = np.zeros((maze_size, maze_size), dtype=bool)
            r_start = max(row - half, 0)
            r_end = min(row + half + 1, maze_size)
            c_start = max(col - half, 0)
            c_end = min(col + half + 1, maze_size)
            visible_mask[r_start:r_end, c_start:c_end] = True

            # Dim everything not in the visible area (if partial obs)
            dimmed = (rgb_maze * dim_factor).astype(np.uint8)
            rgb_maze = np.where(visible_mask[:, :, None], rgb_maze, dimmed)
        
        return rgb_maze.astype(np.uint8)

    def crop_frame(self, frame):
        '''
            Expects CxHxW frame input 
        '''

        half = self.crop_size // 2

         # pad : 1 for walls, 0 otherwise
        padded = np.stack([np.pad(ch, half, constant_values=(1 if i == 0 else 0)) for i, ch in enumerate(frame)],axis=0)

        # Offset pos cause padding
        row,col = self.agent_pos + half
        return padded[:, row-half:row+half+1, col-half:col+half+1]

    def get_obs(self):
        # return self.current_maze

        if self.use_visited : 
            visited_plane = (self.visited > 0).astype(np.float32)

            # single 4-channel frame
            frame = np.concatenate(
                [self.current_maze.copy(), visited_plane[None, ...]],  # (4, H, W)
                axis=0
            )
        else : 
            frame = self.current_maze.copy()

        # Gives the past n states 
        self.previous_states.append(frame)
        self.previous_states = self.previous_states[-self.n_previous_states:]

        if self.partial_obs : 
            cropped_frames = [self.crop_frame(frame) for frame in self.previous_states]
            return np.concatenate(cropped_frames, axis=0).astype(np.float32)

        return np.concatenate(self.previous_states, axis=0).astype(np.float32)

    def shortest_path_length(self):
        '''
        Computes shortest path length using BFS.

        Output: number of steps to goal from agent pos, -1 if unreachable
        '''
        walls = self.current_maze[0]
        N = walls.shape[0]
        
        visited = np.zeros_like(walls, dtype=bool)
        queue = []
        
        start = tuple(self.agent_pos)
        goal = tuple(self.target_pos)

        queue.append((start[0], start[1], 0))
        visited[start] = True

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

        while queue:
            r, c, dist = queue[0]
            queue = queue[1:] 
            
            if (r, c) == goal:
                return dist
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < N and 0 <= nc < N and not visited[nr, nc] and walls[nr, nc] == 0):
                    visited[nr, nc] = True
                    queue.append((nr, nc, dist + 1))

        return -1

    def manhattan_distance(self):
        '''
        Computes Manhattan distance between agent and goal. 
        Output : int
        '''

        return np.abs(self.agent_pos - self.target_pos).sum()

    def dist_from_goal(self):
        return self.shortest_path_length()
    
    def apply_action(self, action):
        '''
        If action is valid, applies action to environment (move in direction) and return 0
        If action invalid, return -1, return -0.5 if new pos has been visited before, 0 otherwise
        '''

        direction = self.action_to_direction[action]

        new_agent_pos = self.agent_pos + direction

        r, c = new_agent_pos
        H, W = self.current_maze.shape[1:]

        # Invalid index
        if not (0 <= r < H and 0 <= c < W):
            return -1 

        # Technically should only need to check for wall collision since the agent is inside the maze
        if self.current_maze[0][tuple(new_agent_pos)] == 1 : 
            return -1
        
        # Move the agent
        self.current_maze[1][tuple(self.agent_pos)] = 0
        self.current_maze[1][tuple(new_agent_pos)] = 1
        
        self.agent_pos = new_agent_pos

        self.visited[tuple(new_agent_pos)] += 1 

        # Penalize revisits
        reward = 1 - np.clip(a_min=0,a_max=2, a=0.5 * self.visited[tuple(new_agent_pos)])
        
        return reward

    def step_and_get_rewards(self, action):
        ''' 
        Applies to action to the environment and returns next state as well as reward
        action : integer [0,1,2,3] -> 0 = go up, 1 = go down, 2 = go left, 3 = go right
        reward (for st+1) : 
            - solve maze : 10
            - impossible move (move into wall) : -1
            - negative distance from goal (manhattan or shortest path)
        '''

        # Checks if correct move and updates the maze
        reward = -0.01 + self.apply_action(action) # -0.01 is penalty for moving, to penalize longer solving

        # Get next obs
        next_obs = self.get_obs()

        new_goal_dist = self.dist_from_goal()
        dist_change = self.curr_dist_goal - new_goal_dist

        done = (new_goal_dist == 0)

        reward += done * 10 + 0.5 * dist_change # Reward at completion

        self.curr_dist_goal = new_goal_dist
        return next_obs, reward, done

    def step(self,action):
        
        self.curr_ep_steps += 1
        
        next_obs, reward, done = self.step_and_get_rewards(action)

        truncated = (self.curr_ep_steps >= self.max_ep_len)

        info = {}
        return next_obs, reward, done, truncated, info
    
    def save_episode_gif(self, filename = "maze_episode", policy = None, fps = 5, scale = 20, recurrent=False):
        '''
        Runs one full episode on a random maze,
        records each rendered frame, and saves as a GIF.
        '''

        for index,maze in enumerate(self.mazes) : 
            obs, _ = self.reset(maze=maze.copy())

            frames = []
            done = False
            
            while not done:

                frame = self.render()
                frames.append(frame)

                # pick an action
                if policy is None:
                    action = self.action_space.sample()
                else:
                    action, _ = policy.predict(obs, deterministic=True)

                obs, reward, done, truncated, info = self.step(int(action))
                done = done or truncated

            # Upsample each one
            big_frames = []
            for f in frames:
                pil = Image.fromarray(f)
                big = pil.resize((f.shape[1]*scale, f.shape[0]*scale), Image.NEAREST)
                big_frames.append(np.array(big))

            # imageio expects list of H×W×3
            imageio.mimsave(filename + f"_{index}.gif", frames, fps=fps)

            if policy is None : 
                break
        
        print(f"Saved episode GIF to {filename}")

    def show_episode(self, policy = None, fps = 5, scale = 20, recurrent=False, random_maze=False, wipe = False):
        
        '''
        Runs one full episode on a random maze, collects frames and display it (notebook)
        '''

        if random_maze : 
            maze = self.mazes[np.random.randint(len(self.mazes))].copy()
        else :
            maze = self.current_maze
            
            # Reset agent pos to start pos
            maze[1][tuple(self.agent_pos)] = 0
            maze[1][tuple(self.start_pos)] = 1

        obs, _ = self.reset(maze=maze)

        frames = []
        done = False

        lstm_states = None

        first_step = True

        while not done:
            frames.append(self.render())

            if policy is None:
                action = self.action_space.sample()
            else:
                if recurrent : 
                    action, lstm_states = policy.predict(obs,state=lstm_states,episode_start=np.array([first_step]), deterministic=True)
                else : 
                    action, _ = policy.predict(obs, deterministic=True)

            first_step = False or wipe
            obs, reward, done, truncated, info = self.step(int(action))
            done = done or truncated

        # Upsample each one
        big_frames = []
        for f in frames:
            pil = Image.fromarray(f.astype(np.uint8))
            big = pil.resize((f.shape[1]*scale, f.shape[0]*scale), Image.NEAREST)
            big_frames.append(np.array(big))

        buf = BytesIO()
        imageio.mimsave(buf, big_frames, format='GIF', fps=fps)
        buf.seek(0)

        display(IPImage(data=buf.getvalue()))

    def test_policy(
        self,
        policy=None,
        recurrent: bool = True,
        thinking_steps: int = 1,
    ):
        '''
            Evaluate policy on every maze stored in self.mazes.
            returns : 
                avg_reward : float
                avg_steps  : float
                success_rate : float
        '''
        total_rewards = []
        total_steps = []
        successes = 0

        for maze in self.mazes:
            obs, _ = self.reset(maze=maze)

            done = False
            lstm_states = None
            first_step = True
            ep_reward = 0.0

            while not done:
                # (k-1) “thinking” passes on the same frame
                if policy is not None and thinking_steps > 1:
                    for _ in range(thinking_steps - 1):
                        if recurrent:
                            _, lstm_states = policy.predict(
                                obs,
                                state=lstm_states,
                                episode_start=np.array([False]),
                                deterministic=True,
                            )
                        else:
                            policy.predict(obs, deterministic=True)

                if policy is None:
                    action = self.action_space.sample()
                else:
                    if recurrent:
                        action, lstm_states = policy.predict(
                            obs,
                            state=lstm_states,
                            episode_start=np.array([first_step]),
                            deterministic=True,
                        )
                    else:
                        action, _ = policy.predict(obs, deterministic=True)

                first_step = False
                obs, reward, done, truncated, _ = self.step(int(action))
                ep_reward += reward
                if done:
                    successes += 1
                done = done or truncated

            total_rewards.append(ep_reward)
            total_steps.append(self.curr_ep_steps)

        total_rewards = np.asarray(total_rewards, dtype=np.float32)
        total_steps = np.asarray(total_steps, dtype=np.float32)
        success_rate = 100.0 * successes / len(self.mazes)

        print(f"Average reward : {total_rewards.mean():.3f}")
        print(f"Average steps  : {total_steps.mean():.1f}")
        print(f"Success rate   : {success_rate:.1f}%")

        return total_rewards.mean(), total_steps.mean(), success_rate
