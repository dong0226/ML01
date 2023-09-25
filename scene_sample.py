import os
import pybullet as p 
import pybullet_data as pd 
from math import pi
import gym
from gym import error,spaces,utils
from gym.utils import seeding
import numpy as np

import pkgutil
egl = pkgutil.get_loader('eglRenderer')
from time import sleep, time

import matplotlib.pyplot as plt

class PushEnv(gym.Env):
    def __init__(self):
        # p.connect(p.GUI)/
        p.connect(p.GUI)
        # p.connect(p.DIRECT)
        
        
        self.action_space = spaces.Box(low=np.ones(3)*-1, high=np.ones(3)) # eef vel: [vx, vy, vz]
        self.observation_space = gym.spaces.Dict({
            'cam_img': gym.spaces.Box(low=0, high=255, shape=(480, 480, 3), dtype=np.uint8),
            'eef_pos': gym.spaces.Box(low=np.ones(3)*-1, high=np.ones(3), dtype=np.float32)
        })
        
        
        dt = 1./20.
        p.setTimeStep(dt)
        self.dv = 5 * dt
        self.visPitch = -50
        self.visYaw = 50
        self.visDis = 1.5
        self.visTargetPos = [0.5, 0, 0]
    
        p.resetDebugVisualizerCamera(cameraDistance=self.visDis, 
                                     cameraYaw=self.visYaw, 
                                     cameraPitch=self.visPitch,
                                     cameraTargetPosition=self.visTargetPos)
        
        p.setAdditionalSearchPath(pd.getDataPath())
        
    def reset(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        
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
        # planeId = p.loadURDF("plane.urdf",  basePosition=[0, 0, -0.63])
        
        # initialize joint position
        rest_angles = [0, -pi/4, 0, -3*pi/4, 0, pi/2, pi/4]
        for i, angle in enumerate(rest_angles):
            p.resetJointState(self.pandaUid, i, angle)
            
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        
    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        
        # --------- robot control ----------
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
        
        # --------- reward calculation ----------
        reward = 0
        obs = None
        done = False
        info = None
        
        # get the object's position and orientation
        ObjState = p.getBasePositionAndOrientation(self.objUid) 
        ObjPosition = ObjState[0]
        ObjOrientation = ObjState[1] # the orientation is in quaternion form
        
        # get contact information
        contact_pts = p.getContactPoints(self.pandaUid, self.objUid, 8, -1) # link 8 and base link
        if contact_pts:
            print("--", contact_pts)
        
        img = self.render()
        # plt.imshow(img)
        # plt.show()
        
        # PLEASE modify the reward function in this part 
        
        if False: # set the terminal condition
            done = True
        
        
        return obs, reward, done, info
        
    def render(self):
        H, W = 240, 240
        view_matrix=p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=self.visTargetPos,
                                                        distance=self.visDis,
                                                        yaw=self.visYaw,
                                                        pitch=self.visPitch,
                                                        roll=0,
                                                        upAxisIndex=2)
        
        proj_matrix=p.computeProjectionMatrixFOV(fov=60,aspect=float(W)/H,
                                                 nearVal=0.1,
                                                 farVal=100.0)
        
        (_,_,px,_,_)=p.getCameraImage(width=W,
                                      height=H,
                                      viewMatrix=view_matrix,
                                      projectionMatrix=proj_matrix,
                                      renderer=p.ER_BULLET_HARDWARE_OPENGL, 
                                      flags=p.ER_NO_SEGMENTATION_MASK)
        
        rgb_array=np.array(px,dtype=np.uint8)
        # rgb_array=np.reshape(rgb_array,(H,W,4))
        rgb_array=rgb_array[:, :, :3]

        return rgb_array
        
    def close(self):
        p.disconnect()

 
if __name__ == "__main__":
    env = PushEnv()
    env.reset()
    
    i = 0

    t1 = time()
    while i < 20*1000:
        env.step([0, 0, -1*0])
        i += 1
        
    print(f"\n\nsimulated 10 seconds in {time() - t1} seconds\n\n")
