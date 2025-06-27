import numpy as np

def make_maze(maze_size):
    '''
    '''

    maze = []

    # Initial walled maze
    for row_id in range(maze_size):
        row = []
        for col_id in range(maze_size):
            # cell = 0 : Up , 1 : Left , 2 : Right , 3 : Bottom ; 4 : row index, 5 : col index,  6 : visited
            row.append([1,1,1,1,row_id,col_id,0])

        maze.append(row)

    # Cells
    stack = [maze[0][0]]
    it = 0
    while len(stack) != 0 and it < 100_000: 

        row_id = stack[-1][4]
        col_id = stack[-1][5]
        
        # Set current cell to visited
        maze[row_id][col_id][6] = 1

        valid_directions = []

        # Bottom
        if row_id + 1 < maze_size : 
            # Check visited
            if maze[row_id + 1][col_id][6] == 0 : 
                valid_directions.append((0,1))
        # Top
        if row_id - 1 >= 0 : 
            if maze[row_id - 1][col_id][6] == 0 : 
                valid_directions.append((0,-1))
        # Left
        if col_id + 1 < maze_size : 
            if maze[row_id][col_id + 1][6] == 0 : 
                valid_directions.append((1,0))
        # Right
        if col_id - 1 >= 0 : 
            if maze[row_id][col_id - 1][6] == 0 : 
                valid_directions.append((-1,0))
        
        if len(valid_directions) > 0 : 
            rand_index = np.random.randint(low=0,high=len(valid_directions))
            
            next_dir = valid_directions[rand_index]

            next_row_id = row_id + next_dir[1]
            next_col_id = col_id + next_dir[0]
            
            if next_dir[0] != 0 :
                # Go right
                if next_dir[0] > 0 : 
                    # Current cell's right
                    maze[row_id][col_id][2] = 0

                    # Next cell's left
                    maze[next_row_id][next_col_id][1] = 0

                else : 
                    # Current cell's left
                    maze[row_id][col_id][1] = 0

                    # Next cell's right
                    maze[next_row_id][next_col_id][2] = 0

            else : 
                # Go bot
                if next_dir[1] > 0 : 
                    # Current cell's bot
                    maze[row_id][col_id][3] = 0

                    # Next cell's top
                    maze[next_row_id][next_col_id][0] = 0

                else : 
                    # Current cell's top
                    maze[row_id][col_id][0] = 0

                    # Next cell's bot
                    maze[next_row_id][next_col_id][3] = 0
            
            stack.append(maze[next_row_id][next_col_id])

        else : 
            stack = stack[:-1]

        it += 1 

    # Create a maze with proper integers for wall/floor/player/goal and then separate into 4 channels
    return separate_maze(expand_maze(maze))

def expand_maze(maze, wall_prob = 0.1):
    '''
    Input : maze array [maze_size x maze_size], each cell is [Wall Up, bottom, left, right, cell row id, cell col id, visited]
    Output : Expanded grid maze, 0 = clear tile, 1 = wall, 2 = current agent position, 3 = target position
    wall_prob : probability of removing a wall 
    '''
    maze_size = len(maze)
    new_maze = np.ones(shape=(maze_size * 2 + 1,maze_size * 2 + 1))

    for i in range(maze_size * 2 + 1):
        for j in range(maze_size * 2 + 1):
            if i % 2 != 0 and j % 2 != 0 :
                new_maze[i][j] = 0
                random_wall_prob = np.random.rand(4)

                # Up
                if maze[i//2][j//2][0] == 0 or random_wall_prob[0] < wall_prob :
                    # Check for border (in case of random wall)
                    if i-1 != 0 : 
                        new_maze[i-1][j] = 0
                # Left
                if maze[i//2][j//2][1] == 0 or random_wall_prob[1] < wall_prob :
                    if j-1 != 0 :
                        new_maze[i][j-1] = 0
                # Right
                if maze[i//2][j//2][2] == 0 or random_wall_prob[2] < wall_prob :
                    if j+1 < maze_size * 2 : 
                        new_maze[i][j+1] =  0
                # Bottom
                if maze[i//2][j//2][3] == 0 or random_wall_prob[3] < wall_prob :
                    if i + 1 < maze_size * 2 :
                        new_maze[i+1][j] = 0

    # Pick starting position (on the left)
    start_row = np.random.randint(low=0,high=maze_size) * 2 + 1
    new_maze[start_row][1] = 2

    # Pick target position (other wall)
    wall_choice = np.random.choice(['top', 'bottom', 'right'])

    if wall_choice == 'top':
        col = np.random.randint(low=0, high=maze_size) * 2 + 1
        new_maze[0][col] = 3
    elif wall_choice == 'bottom':
        col = np.random.randint(low=0, high=maze_size) * 2 + 1
        new_maze[-1][col] = 3
    elif wall_choice == 'right':
        row = np.random.randint(low=0, high=maze_size) * 2 + 1
        new_maze[row][-1] = 3

    return new_maze

def separate_maze(maze):
    '''
    Turns the maze into multiple channels (one-hot) : Walls , Floors, Agent position, Target position
    Input : [maze_size x maze_size] grid maze, 0 = clear tile, 1 = wall, 2 = current agent position, 3 = target position
    Output : 4 x maze_size x maze_size of [0,1]
    '''

    floors = (maze == 0).astype(np.uint8)
    walls = (maze == 1).astype(np.uint8)
    agent_pos = (maze == 2).astype(np.uint8)
    target_pos = (maze == 3).astype(np.uint8)

    new_maze = np.stack([walls, agent_pos, target_pos],axis=0)
    # new_maze = np.stack([walls, floors, agent_pos, target_pos],axis=0)

    return new_maze

def maze_to_rgb(maze):

    maze_size = maze.shape[1]
    rgb_maze = np.zeros((maze_size, maze_size, 3), dtype=np.uint8)
    
    # walls, floors, agent, target = maze
    walls, agent, target = maze
    
    # rgb_maze[floors == 1] = [255, 255, 255]
    rgb_maze[walls == 1] = [0, 0, 0]
    rgb_maze[walls == 0] = [255, 255, 255]
    rgb_maze[target == 1] = [0, 255, 0]
    rgb_maze[agent == 1] = [0, 0, 255]
    
    return rgb_maze