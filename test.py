import object_pushing
import gymnasium as gym
from stable_baselines3 import SAC, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3.td3.policies import MultiInputPolicy as TD3MultiInputPolicy

from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
import os
import numpy as np
import matplotlib.pyplot as plt

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

env_id = "ObjectPushing-v0"
# import pybullet_envs
# Create the env

env = gym.make(env_id)
env = Monitor(env, log_dir)

# env = make_vec_env(env_id, n_envs=2)
# # env = VecNormalize(env)
# env = VecMonitor(env, log_dir)



n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions), dt=1/20, theta=0.6)

model = SAC(env=env, policy=MultiInputPolicy, buffer_size=100000, batch_size=512, verbose=1, device="cuda")#, tensorboard_log="sac_log")
# model = TD3(env=env, policy=TD3MultiInputPolicy, buffer_size=200000, batch_size=512, action_noise=action_noise, verbose=1, device="cuda")

# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, verbose=0)

from tqdm import tqdm

# plt.ion()
for iter, i in enumerate(tqdm(range(10_000))):
   model.learn(2000, log_interval=500, callback=callback)
   model.save("sac-pushing-v0")
  #  model.save("ddpg-pushing-v0")

  #  plt.clf()
  #  plot_results([log_dir], (iter+1)*2000, results_plotter.X_TIMESTEPS, "TD3 LunarLander")
  #  plt.pause(0.01)

# plt.ioff()s



