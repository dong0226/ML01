'''
This is the test code of the Actor-Critic network.
 
'''
import object_pushing
import gymnasium as gym
from pynput import keyboard
from time import sleep
import pybullet as p
from models import ACNet
import torch 
import torch.nn
from torch.optim import Adam
from scene_sample import PushEnv
import numpy as np

# optimizer = Adam()
device = "cuda" if torch.cuda.is_available() else "cpu"
max_speed = 1
FPS = 20
lr = 1e-3



def obs_to_tensor(obs_list, dev):
    '''
    convert observations from numpy array to PyTorch tensors
    '''
    tensor_list = [torch.from_numpy(obs).unsqueeze(0).to(dev) for obs in obs_list]
    return tensor_list

# Define the environment and the actor-critic network
ac_net = ACNet(test=True)
ac_net.to(device)
optimizer = Adam(ac_net.parameters(), lr=lr)
env = PushEnv()
env.reset()

done = False
vel = [0, 0]
while True:
    if done:
        env.reset()
    
    optimizer.zero_grad()
    # Take a step in the environment with the specified velocities
    obs, reward, done,_, info = env.step(vel)
    
    # put obervations into list
    obs_list = [
        obs["cam_img"].transpose(2, 0, 1) / 255, 
                obs["eef_pos"],
                obs["in_touch"], 
                obs["obj_pos"], 
                obs["goal_pos"]
                ]

    action, value, loss = ac_net(obs_to_tensor(obs_list, device), 
                                 target_v=torch.Tensor([[1]]).to(device), 
                                 reward=torch.Tensor([[reward]]).to(device))
    
    
    vel = action[0].detach().cpu().numpy()
    vel = np.clip(vel, -1, 1)

    # optimize the network
    loss.backward()
    optimizer.step()
    
    print("action: ", vel)
    print("value: ", value[0, 0].detach().cpu().numpy(), "loss: ", loss.detach().cpu().numpy())
    
    
    if info['success']:
        print('You have reached the goal!')
    if info['toppled']:
        print('toppled')
    if info['dropped']:
        print('dropped')