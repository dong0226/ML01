import object_pushing
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy

env_id = "ObjectPushing-v0"
# import pybullet_envs
# Create the env

env = gym.make(env_id)

model = SAC(env=env, policy=MultiInputPolicy, buffer_size=10000, verbose=1, device="cuda")

from tqdm import tqdm

for i in tqdm(range(10000)):
   model.learn(1_000, log_interval=100)
   model.save("sac-PandaReachDense-v3")
   env.save("vec_normalize.pkl")


