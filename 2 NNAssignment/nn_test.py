'''
This is the test code of the PushEnv.
 
To test the environment, 
press either the up, down, left or right keys on the keyboard, 
and observe how the Panda robot interact with the environment. 
'''
import object_pushing
import gymnasium as gym
from pynput import keyboard
from time import sleep
import pybullet as p
from models import ACNet
import torch 
import torch.nn
from scene_sample import PushEnv

device = "cuda" if torch.cuda.is_available() else "cpu"
max_speed = 1
FPS = 20

# generate the velocity vector based on the keys pressed
def get_vel_from_keys(keys):
    vx, vy = 0, 0
    if p.B3G_UP_ARROW in keys:
        vx = -max_speed
    elif p.B3G_DOWN_ARROW in keys:
        vx = max_speed
    if p.B3G_LEFT_ARROW in keys:
        vy = -max_speed
    elif p.B3G_RIGHT_ARROW in keys:
        vy = max_speed
        
    return [vx, vy]

def obs_to_tensor(obs_list):
    tensor_list = [torch.from_numpy(obs).unsqueeze(0) for obs in obs_list]
    return tensor_list

# Define the environment
ac_net = ACNet(test=True)
env = PushEnv()
env.reset()

done = False
vel = [0, 0]
while True:
    if done:
        env.reset()
    
    # # Take a step in the environment with the specified velocities
    obs, reward, done,_, info = env.step(vel)
    # ac_net()
    obs_list = [
        obs["cam_img"].transpose(2, 0, 1) / 255, 
                obs["eef_pos"],
                obs["in_touch"], 
                obs["obj_pos"], 
                obs["goal_pos"]
                ]
    # print(obs_list)
    action, value = ac_net(obs_to_tensor(obs_list))
    vel = action[0].detach().numpy()
    print(action[0], value)
    
    
    if info['success']:
        print('You have reached the goal!')
    if info['toppled']:
        print('toppled')
    if info['dropped']:
        print('dropped')