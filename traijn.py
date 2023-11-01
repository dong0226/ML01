
# import
import gym
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import device as torch_device, optim

from models import ACNet            # personal network
from scene_sample import PushEnv    # personal environment
import easydict

# Check local device list
device_list  = [torch.device('cuda' if torch.cuda.is_available() else 'cpu')]
device_names = [str(device) for device in device_list]

print(device_list)

# Make sure at least one device is available
assert len(device_list) > 0

    


# A2C algorithm
class A2C:
    def __init__(self,n_states,n_actions,cfg) -> None:
        self.gamma      = cfg.gamma
        self.device     = cfg.device
        self.model      = cfg.net_name(cfg.embd_dim,cfg.img_conv_list,cfg.img_fc_list,cfg.eef_pos_embd_list,
                                  cfg.policy_net_list,cfg.value_net_list,cfg.test,cfg.obj_pos_embd_list,
                                  cfg.goal_pos_embd_list,cfg.add_noise).to(cfg.device)
        self.optimizer = optim.Adam(self.model.parameters())

    def compute_returns(self,next_value,rewards,masks):
        R       = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step]+ step.gamma*R*masks[step]
            returns.insert(0,R)
        return returns
    
# 测试环境的函数
def test_env(env,model,vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(cfg.device)
        policy, _ = model(state)
        next_state, reward, done, _ = env.step(policy.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward

# 计算回报的函数
def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

# 训练部分
def train(cfg,env):
    # 打印信息
    print('Start training')
    print(f'Env:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}')
    # 加载环境
    scene_sample=PushEnv()
    scene_sample.reset()

    # 环境交互
    n_states = scene_sample.observation_space.shape[0]   # 状态的数量
    n_actions = scene_sample.action_space.n

    # network 导入
    model = ACNet(cfg.embd_dim,cfg.img_conv_list,cfg.img_fc_list,
                  cfg.eef_pos_embd_list,cfg.policy_net_list,cfg.value_net_list,
                  cfg.test,cfg.obj_pos_embd_list,
                  cfg.goal_pos_embd_list,cfg.add_noise)
    # 优化器
    optimizer = optim.Adam(model.parameters())

    step_idx    = 0     # 初始化时间步
    test_rewards = []   # 存储每个测试周期（episode）的累积奖励
    test_ma_rewards = []# 存储每个测试周期（episode）的平均奖励   
    

    while step_idx < cfg.max_steps:

        log_probs = []  # 存储每个时间步的动作选择的对数概率
        values    = []  # 存储每个时间步的状态值的估计
        rewards   = []  # 存储每个时间步的奖励值
        masks     = []

        # rollout 连续动作
        for _ in range(cfg.n_steps):
            # 将状态转换为PyTorch张量并传递给神经网络
            state_tensor = torch.FloatTensor(n_states,n_actions).to(cfg.device)
            policy, value, loss = model(state_tensor)
            # 从策略中采样动作
            action = policy.sample()
            # 计算对数概率并存储
            log_prob = policy.log_prob(action)
            log_probs.append(log_prob)
            # 执行动作并观察环境
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            # 存储值函数估计和奖励
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(cfg.device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(cfg.device))
            step_idx +=1
            # 更新熵值
            entropy += policy.entropy().mean()
            # 每200步进行一次测试
            if step_idx % 200 == 0:
                test_reward = np.mean([test_env(env, model) for _ in range(10)])
                print(f"step_idx:{step_idx}, test_reward:{test_reward}")
                test_rewards.append(test_reward)

                if test_ma_rewards:
                    test_ma_rewards.append(0.9 * test_ma_rewards[-1] + 0.1 * test_reward)
                else:
                    test_ma_rewards.append(test_reward)
        
        # 更新当前状态
        next_state = torch.FloatTensor(next_state).to(cfg.device)
        _, next_value = model(next_state)
        returns = compute_returns(next_value, rewards, masks)
        log_probs = torch.cat(log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(values)

        optimizer.zero_grad()
        loss.backward()


# 相关参数
cfg = easydict.EasyDict({
    "algo_name" : 'A2C',
    "env_name"  : 'scene_sample',
    "max_steps" : 300,
    "n_step"    : 50,
    "gamma"     : 0.99,
    "device"    : torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"),
    "lr"        : 1e-3,
    "net_name"  : 'models',

    #环境的各状态的维度和张量
    "embd_dim": 16,
    "img_conv_list": [4, 'P', 8, 'P', 16],
    "img_fc_list": [64, 16],
    "eef_pos_embd_list": [16],
    "policy_net_list": [32, 2],
    "value_net_list": [32, 1],
    "test": False,
    "obj_pos_embd_list": [16],
    "goal_pos_embd_list": [16],
    "add_noise": True
})



if __name__ == '__main__':
    scene_sample=PushEnv(use_camera=True, use_gui=False)
    train(cfg,scene_sample)