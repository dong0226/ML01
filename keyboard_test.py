import object_pushing
import gymnasium as gym
from pynput import keyboard
from time import sleep
import pybullet as p

max_speed = 1
FPS = 20

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

# Define the environment
env = gym.make('ObjectPushing-v0')
env.reset()

done = False
while True:
    if done:
        env.reset()
        
    keys = p.getKeyboardEvents()
    
    vel = get_vel_from_keys(keys)
    # # Take a step in the environment with the specified velocities
    obs, reward, done,_, info = env.step(vel)
    sleep(1/FPS)
    
    if info['success']:
        print('You have reached the goal!')
    if info['toppled']:
        print('toppled')
    if info['dropped']:
        print('dropped')