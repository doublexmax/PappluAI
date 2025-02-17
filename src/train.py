# filepath: /home/maxxt/PappluAI/src/train.py
import numpy as np
from environment import PappluEnv

def train():
    env = PappluEnv()
    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = np.random.choice(['pick', 'discard'])  # Placeholder for action selection
            next_state, reward, done = env.step(action)
            # Update model with (state, action, reward, next_state)
            state = next_state

if __name__ == "__main__":
    train()