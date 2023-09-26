from gymnasium.envs.registration import register

register(
    id='ObjectPushing-v0', 
    entry_point='object_pushing.envs:PushEnv', 
    max_episode_steps=400,
)