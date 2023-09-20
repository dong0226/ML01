import os
import pybullet as p 
import pybullet_data as pd 
from math import pi

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())

tableUid = p.loadURDF("models/table/table.urdf",basePosition=[0.5,0,-0.63])
objUid = p.loadURDF("models/random/010.urdf", 
                    basePosition=[0.5, 0, 0.1], 
                    baseOrientation=p.getQuaternionFromEuler([pi/2, 0, 0]), 
                    globalScaling=2)
GoalUid = p.loadURDF("models/block.urdf",basePosition=[0.8,-0.3, 0])

p.setGravity(gravX=0, gravY=0, gravZ=-9.81) # set gravity

# load the panda robot and set its joint angles 
pandaUid = p.loadURDF("models/franka_panda/panda.urdf",useFixedBase=True)
p.setJointMotorControl2(pandaUid,0,p.POSITION_CONTROL,0)
p.setJointMotorControl2(pandaUid,1,p.POSITION_CONTROL,-pi/4.)
p.setJointMotorControl2(pandaUid,2,p.POSITION_CONTROL,0)
p.setJointMotorControl2(pandaUid,3,p.POSITION_CONTROL,-3*pi/4.)
p.setJointMotorControl2(pandaUid,4,p.POSITION_CONTROL,0)
p.setJointMotorControl2(pandaUid,5,p.POSITION_CONTROL,pi/2)
p.setJointMotorControl2(pandaUid,6,p.POSITION_CONTROL,pi/4)


while True:
    p.stepSimulation()