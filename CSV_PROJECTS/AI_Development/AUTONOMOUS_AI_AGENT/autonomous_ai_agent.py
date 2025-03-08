
#       ---- IMPORT_REQUIRED_LIBRARIES ----
# gym: Reinforcement Learning
import gym 
# torch: "Deep Learning": torchvision.models 
# (pre-trained ResNet18)
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
import numpy as np 
# cv2: "Image Processing": (OpenCV)
import cv2
from torchvision import models, transforms
# deque: Memory Buffer
from collections import deque
import random
# gTTS: Text-to-Speech
from gtts import gTTS 
import os
# tkinter: User Interface
import tkinter as tk 
from tkinter import scrolledtext
from threading import Thread


#       ---- LOAD_A_PRETRAINED_MODEL_FOR_OBJECT_CLASSIFICATION_CLASS ----
# Image classification using "ResNet".add()
# Uses a pre-trained ResNet18 model.
# Transforms input images to match ImageNet format.
# Returns predicted class index.
class ObjectClassifier:
    def __init__(self):
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def classify(self, image):
        image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(image)
        return torch.argmax(output).item()
    
    
    
#       ---- DEEP_Q_NETWORK_FOR_NAVIGATION_CLASS ----
# Defines a 3-layer fully connected neural network.
# Inputs: State of the environment.
# Outputs: Q-values for each action.
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

#       ---- EXPERIENCE_REPLAY_BUFFER_CLASS_(REPLAY_BUFFER) ----
# Stores past experiences (state, action, reward, next_state, done).
# Breaks correlation in training by randomly sampling past experiences.
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    
    def size(self):
        return len(self.buffer)
    
    
    
#       ---- TRAINING_FUNCTION_FOR_DQN ----
# Samples a batch from the replay buffer.
# Computes Q-values and target Q-values.
# Uses MSE loss to update the model.
def train_dqn(agent, target_agent, memory, optimizer, batch_size, gamma):
    if memory.size() < batch_size:
        return
    batch = memory.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
    
    
    q_values = agent(states).gather(1, actions)
    with torch.no_grad():
        max_next_q_values = target_agent(next_states).max(1, keepdim=True)[0]
        target_q_values = rewards + (gamma * max_next_q_values * (1 - dones))
    loss = F.mse_loss(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
#       ---- AI_AGENT_CLASS_(MAIN_RL_AGENT) ----
# Defines DQN agent for CartPole-v1.
# Uses ε-greedy exploration.
# # Uses "e-greedy (ε-greedy) policy" for exploration. 
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
class AutonomousAgent:
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n 
        self.agent = DQN(self.state_dim, self.action_dim)
        self.target_agent = DQN(self.state_dim, self.action_dim)
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.optimizer = optim.Adam(self.agent.parameters(), lr=0.001)
        self.memory = ReplayBuffer(1000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.gamma = 0.99
        self.batch_size = 64
        self.target_update_freq = 10
        self.classifier = ObjectClassifier()
        
        
        
    def verbalize(self, text):
        tts = gTTS(text=text, lang='en')
        tts.save("temp.mp3")
        os.system("start temp.mp3")
        
        
    def train_agent(self, episodes=500):
        for episode in range(episodes):
            state = self.env.reset()[0]
            state = np.array(state, dtype=np.float32)
            total_reward = 0
            for t in range(200):
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = torch.argmax(self.agent(torch.tensor(state).unsqueeze(0))).item()
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = np.array(next_state, dtype=np.float32)
                self.memory.add(state, action, reward, next_state, done)
                train_dqn(self.agent, self.target_agent, self.memory, self.optimizer, self.batch_size, self.gamma)
                state = next_state
                total_reward += reward
                if done:
                    break
                
                
            if episode % self.target_update_freq == 0:
                self.target_agent.load_state_dict(self.agent.state_dict())  
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            print(f"Episode: {episode}, Total Reward: {total_reward}")
            



#       ---- USER_INTERFACE ----
# Creates a "TKinter GUI" for user feedback
class AgentUI:
    def __init__(self, agent):
        self.agent = agent
        self.window = tk.Tk()
        self.window.title("Autonomous Agent UI")
        self.text_area = scrolledtext.ScrolledText(self.window, width=60, height=20)
        self.text_area.pack()
        self.feedback_entry = tk.Entry(self.window, width=50)
        self.feedback_entry.pack()
        self.submit_button = tk.Button(self.window, text="Submit Feedback", command=self.submit_feedback)
        self.submit_button.pack()
        self.window.mainloop()
        
        
# This prevents the GUI from blocking training
if __name__ == "__main__":
    agent = AutonomousAgent()
    ui = AgentUI(agent)
    Thread(target=agent.train_agent, args=(agent,), daemon=True).start()

        
    def submit_feedback(self):
        feedback = self.feedback_entry.get()
        self.text_area.insert(tk.END, f"User Feedback: {feedback}\n")
        self.feedback_entry.delete(0, tk.END)



    
                   
                    