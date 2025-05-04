import numpy as np

class Environment:
    def __init__(self):
        # Define the grid (4x4)
        # 0 = Start, 1 = Frozen, 2 = Hole, 3 = Goal
        self.state_grid = np.array([
            [0, 1, 1, 1],
            [1, 2, 1, 2],
            [1, 1, 1, 2],
            [2, 1, 1, 3]
        ])
        
        self.n_rows, self.n_cols = self.state_grid.shape
        self.state = (0, 0)  # Start position (row, column)
        self.done = False

        # Actions: 0=left, 1=right, 2=up, 3=down
        self.action_space = [0, 1, 2, 3]
        self.n_actions = len(self.action_space)
        self.n_states = self.state_grid.size

    def reset(self):
        """Reset the environment to initial state.
        
        Returns:
            tuple: The initial state (row, col)
        """
        self.state = (0, 0)
        self.done = False
        return self.state

    def step(self, action):
        """Take a step in the environment.
        
        Args:
            action (int): The action to take (0=left, 1=right, 2=up, 3=down)
            
        Returns:
            tuple: (action, new_state, reward, done)
            - action (int): The actual action taken (may be different due to stochasticity)
            - new_state (tuple): The new state (row, col)
            - reward (float): The reward received
            - done (bool): Whether the episode has ended
        """
        row, col = self.state

        # Apply action (with 20% chance of slipping to a random direction)
        original_action = action
        if np.random.rand() < 0.20:  # Stochastic environment
            action = np.random.choice(self.action_space)

        # Calculate new position
        if action == 0: new_state = (row, max(col-1, 0))          # Left
        elif action == 1: new_state = (row, min(col+1, self.n_cols-1))  # Right
        elif action == 2: new_state = (max(row-1, 0), col)        # Up
        elif action == 3: new_state = (min(row+1, self.n_rows-1), col)  # Down

        # Get cell type
        cell_value = self.state_grid[new_state]

        # Determine reward and termination
        if cell_value == 3:  # Goal
            reward = 10
            done = True
        elif cell_value == 2:  # Hole
            reward = -5
            done = True
        else:  # Frozen
            reward = 0
            done = False

        self.done = done
        self.state = new_state
        return action, new_state, reward, done
    
    def pos_to_one_hot(self, row, col):
        """Convert (row, col) into a one-hot vector of length 16.
        
        Args:
            row (int): The row coordinate
            col (int): The column coordinate
            
        Returns:
            numpy.ndarray: A one-hot vector representation of the state
        """
        state_idx = row * self.n_cols + col  # integer in [0..15]
        one_hot = np.zeros(self.n_states, dtype=np.float32)
        one_hot[state_idx] = 1.0
        return one_hot
    
    def state_to_one_hot(self, state):
        """Convert a state tuple (row, col) into a one-hot vector.
        
        Args:
            state (tuple): The state as (row, col)
            
        Returns:
            numpy.ndarray: A one-hot vector representation of the state
        """
        row, col = state
        return self.pos_to_one_hot(row, col)
    
    def render(self):
        """Render the environment as a string.
        
        Returns:
            str: A string representation of the environment
        """
        grid_chars = np.array([
            ['S', 'F', 'F', 'F'],
            ['F', 'H', 'F', 'H'],
            ['F', 'F', 'F', 'H'],
            ['H', 'F', 'F', 'G']
        ])
        
        result = ""
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if (i, j) == self.state:
                    result += " X "
                else:
                    result += " " + grid_chars[i, j] + " "
            result += "\n"
        return result

if __name__ == "__main__":
    # Simple test of the environment
    env = Environment()
    state = env.reset()
    print("Initial state:", state)
    print(env.render())
    
    # Take a few random actions
    for i in range(5):
        action = np.random.choice(env.action_space)
        actual_action, next_state, reward, done = env.step(action)
        print(f"Action: {action}, Actual action: {actual_action}, New state: {next_state}, Reward: {reward}, Done: {done}")
        print(env.render())
        if done:
            break
