import os
import pybullet as p 
import pybullet_data as pd 
from math import pi
import gymnasium as gym

from gymnasium import error,spaces,utils
from gymnasium.utils import seeding
import numpy as np

import pkgutil
egl = pkgutil.get_loader('eglRenderer')
from time import sleep, time

import matplotlib.pyplot as plt

class PushEnv(gym.Env):
    def __init__(self, use_camera=False):
        # p.connect(p.GUI)/

        self.use_camera = use_camera

        if use_camera:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        # p.connect(p.DIRECT)
        
        
        self.action_space = spaces.Box(low=np.ones(2)*-1, high=np.ones(2)) # eef vel: [vx, vy, vz]

        if use_camera:
            self.observation_space = gym.spaces.Dict({
                'cam_img': gym.spaces.Box(low=0, high=255, shape=(240, 240, 3), dtype=np.uint8),
                'eef_pos': gym.spaces.Box(low=np.array([0, -0.5, -0.1]) -0.05, 
                                        high=np.array([1, 0.5, 1]) +0.05, 
                                        dtype=np.float32),
                'obj_pos': gym.spaces.Box(low=np.array([0, -0.5, -0.1]) -0.05, 
                                        high=np.array([1, 0.5, 1]) +0.05, 
                                        dtype=np.float32),
                'goal_pos': gym.spaces.Box(low=np.array([0, -0.5, -0.1]) -0.05, 
                                        high=np.array([1, 0.5, 1]) +0.05, 
                                        dtype=np.float32),
            })
        else:
            self.observation_space = gym.spaces.Dict({
                'eef_pos': gym.spaces.Box(low=np.array([0, -0.5, -0.1]) -0.05, 
                                        high=np.array([1, 0.5, 1]) +0.05, 
                                        dtype=np.float32),
                'obj_pos': gym.spaces.Box(low=np.array([0, -0.5, -0.1]) -0.05, 
                                        high=np.array([1, 0.5, 1]) +0.05, 
                                        dtype=np.float32),
                'goal_pos': gym.spaces.Box(low=np.array([0, -0.5, -0.1]) -0.05, 
                                        high=np.array([1, 0.5, 1]) +0.05, 
                                        dtype=np.float32),
                })

        # self.observation_space = gym.spaces.Box(low=0, high=255, shape=(240, 240, 3), dtype=np.uint8)
        
        self.max_step = 200
        self.dt = 1./20.
        p.setTimeStep(self.dt)
        self.dv = 1 * self.dt
        self.visPitch = -50 -10
        self.visYaw = 50 +40
        self.visDis = 1
        self.visTargetPos = [0.5, 0, 0]
        self.z = 0.14
        self.current_step = 0
    
        p.resetDebugVisualizerCamera(cameraDistance=self.visDis, 
                                     cameraYaw=self.visYaw, 
                                     cameraPitch=self.visPitch,
                                     cameraTargetPosition=self.visTargetPos)
        
        p.setAdditionalSearchPath(pd.getDataPath())
        
    def reset(self, seed=None):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)

        p.resetSimulation()
        # self.goal = np.array([0.8, -0.3, 0])
        self.goal = np.random.uniform(low=[0.45, -0.4, 0], high=[0.6, 0.4, 0])
        self.init_pos = np.random.uniform(low=[0.3, -0.35, 0], high=[0.35, 0.35, 0])
        self.distVec = self.goal - self.init_pos

        robot_init_y = np.random.uniform(low=-0.4, high=0.4)

        self.tableUid = p.loadURDF("models/table/table.urdf",basePosition=[0.5,0,-0.62])
        # self.objUid = p.loadURDF("models/random/010.urdf", 
        #                     basePosition=[0.5, 0, 0.05], 
        #                     baseOrientation=p.getQuaternionFromEuler([pi/2, 0, 0]), 
        #                     globalScaling=2)
        self.objUid = p.loadURDF("models/random/cube0.urdf", basePosition=self.init_pos, globalScaling=1)
        self.GoalUid = p.loadURDF("models/block.urdf",basePosition=self.goal)
        

        p.setGravity(gravX=0, gravY=0, gravZ=-9.81) # set gravity

        # load the panda robot and set its joint angles 
        self.pandaUid = p.loadURDF("models/franka_panda/panda.urdf",useFixedBase=True, basePosition=[-0.1, 0, 0])
        # planeId = p.loadURDF("plane.urdf",  basePosition=[0, 0, -0.63])
        
        
        rest_angles = [0, -pi/4, 0, -3*pi/4, 0, pi/2, pi/4]
        for i, angle in enumerate(rest_angles):
            p.resetJointState(self.pandaUid, i, angle)

        # initialize joint position
        rest_angles = p.calculateInverseKinematics(self.pandaUid, 8, [0.25, robot_init_y, self.z], p.getQuaternionFromEuler([0.,-pi,pi/2.]))

        for i, angle in enumerate(rest_angles):
            p.resetJointState(self.pandaUid, i, angle)



        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

        for _ in range(30):
            self.step([0, 0])[0]


        eefPos = np.array(p.getLinkState(self.pandaUid, 8)[0])
        ObjState = p.getBasePositionAndOrientation(self.objUid)
        objPos = np.array(ObjState[0])

        if self.use_camera:
            img = self.render()
            observation = {
                "cam_img": img, 
                "eef_pos": eefPos.astype(np.float32), 
                "obj_pos": objPos.astype(np.float32), 
                "goal_pos": self.goal.astype(np.float32), 

            }
        else:
            observation = {
                "eef_pos": eefPos.astype(np.float32), 
                "obj_pos": objPos.astype(np.float32), 
                "goal_pos": self.goal.astype(np.float32), 

            }


        
        self.current_step = 0
        # observation = img
        return observation, {}
        
    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        
        # --------- robot control ----------
        dx, dy = action
        dx *= self.dv
        dy *= self.dv
        # dz *= self.dv
        
        currentLinkState = p.getLinkState(self.pandaUid, 7) # the pose of the 7th link
        currentPosition = currentLinkState[4]
        currentOrientation = currentLinkState[5]
        
        newPosition=[currentPosition[0]+dx,
                     currentPosition[1]+dy,
                     self.z]
        
        currentJointAngles = p.getJointState(self.pandaUid, 7)
        newJointAngles = p.calculateInverseKinematics(self.pandaUid, 7, newPosition, p.getQuaternionFromEuler([0.,-pi,pi/2.]))
        
        # set new joint values
        for i, angle in enumerate(newJointAngles):
            p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, angle)
        
        p.stepSimulation()
        self.current_step += 1
        
        # --------- get observation ----------
        reward = -0.1 * self.dt # initialize the reward with the time penalty
        observation = {}
        done = False
        info = {}
        
        # get the object's position and orientation
        ObjState = p.getBasePositionAndOrientation(self.objUid)

        objVel = np.array(p.getBaseVelocity(self.objUid))[0]
        objPos = np.array(ObjState[0])
        objOrn = ObjState[1] # the orientation is in quaternion form
        objEuler = np.array(p.getEulerFromQuaternion(objOrn))

        eefStates = p.getLinkState(self.pandaUid, 8, computeLinkVelocity=1)
        eefPos = np.array(eefStates[0])
        eefVel = np.array(eefStates[6])

        
        
        # get contact information
        contact_pts = p.getContactPoints(self.pandaUid, self.objUid, 8, -1) # link 8 and base link
        isTouch = True if contact_pts else False
        # if contact_pts:
        #     print("--", contact_pts)
        

        if self.use_camera:
            img = self.render()
            observation = {
                "cam_img": img, 
                "eef_pos": eefPos.astype(np.float32), 
                "obj_pos": objPos.astype(np.float32), 
                "goal_pos": self.goal.astype(np.float32), 

            }
        else:
            observation = {
                "eef_pos": eefPos.astype(np.float32), 
                "obj_pos": objPos.astype(np.float32), 
                "goal_pos": self.goal.astype(np.float32), 

            }
        
        # --------- reward calculation ----------

        velReward = self.distVec[:2] @ objVel[:2]
        reward += velReward * 10  #* self.dt * 0.1
        # print(velReward * 100)

        
        
        # add a penalty if the object doesn't move
        if np.linalg.norm(objVel[:2]) <= 0.002:
            reward -= 0.1 * self.dt
            # effDist =  np.linalg.norm(eefPos[:2] - objPos[:2])
            # effDist = 0 if isTouch else (effDist - 0.05)
            effDist = objPos[:2] - eefPos[:2]
            # reward -= 20 * self.dt * (effDist @ )

            reward += 20 * self.dt *  (effDist @ eefVel[:2])


        
        # set the terminal condition
        if abs(eefPos[1]) > 0.5 or eefPos[0] < -0.1 or eefPos[0] > 0.55: # the eef goes outside the working area
            done = True
            if eefPos[0] > 0.55:
                reward -= 5
            else:
                reward -= 2
        elif objPos[2] < -0.1: # the object falls down the table
            done = True
            reward -= 2
        # elif abs(objOrn[0]) : # the object is toppled
            # done = True
            # reward -= 1
        elif np.linalg.norm(objPos[:2] - self.goal[:2]) < 0.1: # the object reaches the goal
            done = True
            reward += 50
        else:
            if self.current_step >= self.max_step:
                done = True

                if isTouch:
                    reward += 2
                else:
                    reward -= 1


        
        # print(reward)
        self.distVec = self.goal - objPos

        # observation = img
        return observation, reward, done, False, info
        
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
        
        (_, _, px, _, _)=p.getCameraImage(width=W,
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
    done = False
    # while i < 20*1000:
    while not done:
        obs, _, done, _, _ = env.step([1, 0])
        i += 1
        # print(obs)
        
    print(f"\n\nsimulated 10 seconds in {time() - t1} seconds\n\n")
