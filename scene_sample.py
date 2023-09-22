import os
import pybullet as p 
import pybullet_data as pd 
from math import pi
import gym
from gym import error,spaces,utils
from gym.utils import seeding
import numpy as np

from time import sleep

class PushEnv(gym.Env):
    def __init__(self):
        p.connect(p.GUI)
        
        self.action_space = spaces.Box(low=np.ones(3)*-1, high=np.ones(3)) # eef vel: [vx, vy, vz]
        self.observation_space = gym.spaces.Dict({
            'cam_img': gym.spaces.Box(low=0, high=255, shape=(480, 480, 3), dtype=np.uint8),
            'eef_pos': gym.spaces.Box(low=np.ones(3)*-1, high=np.ones(3), dtype=np.float32)
        })
        
        self.dv = 10 / 240
        
    def reset(self):
        p.resetSimulation()

        self.tableUid = p.loadURDF("models/table/table.urdf",basePosition=[0.5,0,-0.63])
        self.objUid = p.loadURDF("models/random/010.urdf", 
                            basePosition=[0.5, 0, 0.1], 
                            baseOrientation=p.getQuaternionFromEuler([pi/2, 0, 0]), 
                            globalScaling=2)
        self.GoalUid = p.loadURDF("models/block.urdf",basePosition=[0.8,-0.3, 0])

        p.setGravity(gravX=0, gravY=0, gravZ=-9.81) # set gravity

        # load the panda robot and set its joint angles 
        self.pandaUid = p.loadURDF("models/franka_panda/panda.urdf",useFixedBase=True)
        
        # initialize joint position
        rest_angles = [0, -pi/4, 0, -3*pi/4, 0, pi/2, pi/4]
        for i, angle in enumerate(rest_angles):
            p.resetJointState(self.pandaUid, i, angle)
            

        
    def step(self, action):
        dx, dy, dz = action
        dx *= self.dv
        dy *= self.dv
        dz *= self.dv
        
        currentLinkState = p.getLinkState(self.pandaUid, 7) # the pose of the 7th link
        currentPosition = currentLinkState[4]
        currentOrientation = currentLinkState[5]
        
        newPosition=[currentPosition[0]+dx,
                     currentPosition[1]+dy,
                     currentPosition[2]+dz]
        
        currentJointAngles = p.getJointState(self.pandaUid, 7)
        newJointAngles = p.calculateInverseKinematics(self.pandaUid, 7, newPosition, p.getQuaternionFromEuler([0.,-pi,pi/2.]))
        
        # set new joint values
        for i, angle in enumerate(newJointAngles):
            p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, angle)
        
        p.stepSimulation()
        
        reward = 0
        obs = None
        done = False
        info = None
        
        
        
        return obs, reward, done, info
        
    def render(self):
        pass
        
    def close(self):
        p.disconnect()

 
if __name__ == "__main__":
    env = PushEnv()
    env.reset()
    
    i = 0
    while i < 240:
        env.step([0, 0, -1])
        sleep(1/240)
        
        # i += 1