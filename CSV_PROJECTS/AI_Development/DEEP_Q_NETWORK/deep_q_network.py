
#       ---- IMPORT_REQUIRED_LIBRARIES ----
# "gym": Environment for reinforcement learning
# "torch (PyTorch)": Deep learning frame
# "collections.deque": Memory buffer for experience replay
import gym 
import random
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
from collections import deque


#       ---- CREATE_THE_DEEP_Q_NETWORK_(DQN) ----
class DQN(nn.Module): 
    # What's going on in here:
    # # "Takes in state as input", and outputs 
    # Q-values for each action. 
    # Uses "ReLU activation" for non-linearity.
    # The "last layer outputs Q-values", for 
    # each action. ----
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # First hidden layer
        self.fc1 = nn.Linear(state_dim, 128)
        # Second hidden layer
        self.fc2 = nn.Linear(128, 128)
        # Output layer
        self.fc3 = nn.Linear(128, action_dim)
        
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output raw Q-values
        return self.fc3(x)
    
    
    
#       ---- IMPLEMENT_EXPERIENCE_REPLAY ----
class ReplayBuffer: 
    # What's going on in here:
    # Stores previous experiences "(state, action,
    # reward, next_sate, done)". 
    # "Samples"past experiences to break correlation 
    # in training. ----
    def __init__(self, capacity):
        # Stores past experiences
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done)) 
        
    def sample(self, batch_size):
        # Get random experience
        return random.sample(self.buffer, batch_size)  
    
    def size(self):
        return len(self.buffer) 
    
    
    
#       ---- DEFINE_DQN_TRAINING_LOGIC ----
# What's going on in here: 
# Samples a batch of experience replay. 
# Computes "Q-values and target_q_values". 
# Uses "Mean Squared Error Loss (mse_loss)" to train 
# the model 
def train_dqn(agent, target_agent, memory, optimizer, batch_size, gamma):
    if memory.size() < batch_size:
        # Do nothing. Wait until we have 
        # enough experiences
        return
    
    
    batch = memory.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64). unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
    
    # Compute Q-values for the taken actions
    q_values = agent(states).gather(1, actions)
    
    # Compute target Q-values
    with torch.no_grad():
        max_next_q_values = target_agent(next_states).max(1, keepdim=True)[0]
        target_q_values = rewards + (gamma * max_next_q_values * (1 - dones))
        
    # Compute loss
    loss = F.mse_loss(q_values, target_q_values)
    
    # Update network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
    #       ---- TRAIN_THE_AGENT_IN_OPENAI_GYM ----
    # What's going on in here: 
    # Loads "CartPole-v1". 
    # Initializes "DQN and target network". 
    # Uses "e-greedy (ε-greedy) policy" for exploration. 
    # NOTE: ε(epsilon)-greedy: In the context of reinforcement 
    # learning, the ε-greedy policy is a strategy for balancing 
    # exploration and exploitation. It's commonly used in 
    # algorithms like Q-learning to help an agent learn the optimal 
    # actions in an environment.
    # The ε (epsilon) represents a probability, typically a small 
    # value between 0 and 1. With probability ε, the agent chooses 
    # an action randomly, exploring the environment. With probability 
    # 1-ε, the agent chooses the action that it believes to be the best 
    # based on its current knowledge (exploitation).  
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n 
    
    
    # Initialize the networks
    agent = DQN(state_dim, action_dim)
    target_agent = DQN(state_dim, action_dim)
    # Copy the initial weights
    target_agent.load_state_dict(agent.state_dict())
    
    
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    memory = ReplayBuffer(1000)
    
    
    # Hyperparameters
    episodes = 500
    batch_size = 64
    gamma = 0.99
    # Exploration rate
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    # Update target network every 10 episodes
    target_update_freq = 10
    
    
    
    #       ---- TRAIN_THE_DQN_AGENT ----
    for episode in range(episodes):
        state = env.reset()[0]
        state = np.array(state, dtype=np.float32)
        total_reward = 0
        
        
        # Maximum steps per episode
        # The index "t" tracks the step count
        for t in range(200):
            # Choose action (epsilon-greedy)
            if random.random() < epsilon:
                # Exploration
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    # Exploitation
                    action = torch.argmax(agent(torch.tensor(state).unsqueeze(0))).item()
                    
            # Take action
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            
            # Store experience in replay memory
            memory.add(state, action, reward, next_state, done)
            
            # Train the model
            train_dqn(agent, target_agent, memory, optimizer, batch_size, gamma)
            state = next_state
            total_reward += reward
            
            if done:
                # This break ensures "t" index is 
                # effectively used in the loop
                break
            
            
        # Update target network
        if episode % target_update_freq == 0:
            target_agent.load_state_dict(agent.state_dict())
            
        # Reduce exploration rate
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")
            
                
    
    
    
        



