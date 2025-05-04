import numpy as np
import random
from collections import namedtuple, deque

# Define the experience namedtuple
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size=100000, batch_size=64):
        """Initialize a ReplayBuffer object.
        
        Args:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory.
        
        Args:
            state: Current state
            action (int): Action taken
            reward (float): Reward received
            next_state: State after taking action
            done (bool): Whether the episode is done
        """
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory.
        
        Returns:
            tuple: (states, actions, rewards, next_states, dones)
            - states: Batch of current states
            - actions: Batch of actions taken
            - rewards: Batch of rewards received
            - next_states: Batch of next states
            - dones: Batch of done flags
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # Convert to numpy arrays
        states = np.array([e.state for e in experiences], dtype=np.float32)
        actions = np.array([e.action for e in experiences], dtype=np.int32)
        rewards = np.array([e.reward for e in experiences], dtype=np.float32)
        next_states = np.array([e.next_state for e in experiences], dtype=np.float32)
        dones = np.array([e.done for e in experiences], dtype=np.float32)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory) 