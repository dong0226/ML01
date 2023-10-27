import torch 
import torch.nn as nn
from functools import reduce
import numpy as np

IMAGE_SHAPE = (3, 240, 240)
EEF_POS_SHAPE = 3
OBJ_POS_SHAPE = 3
GOAL_POS_SHAPE = 3

mse_fcn = nn.MSELoss()



def create_conv_layers(in_channels, architecture):
    '''
    Returns a sequential convolutional neural network with the specified architecture.
    
    For example, if in_channels = 3, architecture = [6, 'M', 12], 
    the function will generate a neural network with the following layers:
    {
        nn.Conv2d(3, 6, kernel_size=(3,3), stride=(1,1), padding=(1,1)), 
        nn.ReLU()
        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), )
        nn.Conv2d(6, 12, kernel_size=(3,3), stride=(1,1), padding=(1,1)), 
        nn.ReLU()
    }
    
    Args: 
        in_channels: Integer
        architecture: list (composed with positive integers or the stirng 'M' only)
    
    Returns: 
        A PyTorch sequential model having the specified network architecture
    '''
    layers = []
    
    for x in architecture:
        out_channels = x
        
        if type(x) == int:
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=(3,3), stride=(1,1), padding=(1,1)), nn.ReLU()]
            in_channels = x
        elif x == 'P':
            layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
            
    return nn.Sequential(*layers)


def create_fc_layers(in_features, architecture, last_act=None):
    '''
    This function returns a fully-connected neural network with the given architecture 
    
    For example, if in_channels = 8, architecture = [16, 32, 64], last_act = nn.Sigmoid(), 
    the function will generate a neural network with the following layers:: 
    {
        nn.Linear(8, 16), 
        nn.ReLU(),
        nn.Linear(16, 32), 
        nn.ReLU(),
        nn.Linear(32, 64), 
        nn.Sigmoid()
    }
    
    Args: 
        in_features: Integer
        architecture: list (composed with positive integers)
        last_act: nn.Module()
    
    Returns: 
        A PyTorch sequential model having the specified network architecture
    '''
    layers = []      
    for x in architecture:
        out_features = x
        layers += [nn.Linear(in_features, out_features), nn.ReLU()]
        in_features = x

    layers.pop(-1)
    
    if last_act is not None:
        layers += [last_act]

    return nn.Sequential(*layers)


    
class ConvFCNet(nn.Module):
    '''
    This class provides a flexible and configurable framework for 
    building custom neural networks with convolutional layers 
    followed with linear layers
    
    Attributes: 
        input_dim: input dimension of the image
        conv_layer_list: the list used as the 'architecture' argument for the function create_conv_layers()
        fc_layer_list: the list used as the 'architecture' argument for the function create_fc_layers()
    '''
    def __init__(self, 
                 input_dim,
                 conv_layer_list, 
                 fc_layer_list):
        super().__init__()
        # create the convolutional neural network
        self.in_channels = input_dim[0]
        self.conv_layers = create_conv_layers(self.in_channels, conv_layer_list)
        
        # automatically determine the input dimention of the linear layers
        sample_data = torch.zeros(1, self.in_channels, input_dim[1], input_dim[2])
        sample_out = self.conv_layers(sample_data)
        output_dim = reduce(lambda x, y: x * y, sample_out.shape) # calculate the output size of the created convolution layers
        self.output_dim = output_dim
        
        # create the linear layer
        self.fc_layers = create_fc_layers(output_dim, fc_layer_list)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.conv_layers(x)
        x = self.fc_layers(x.reshape(B, -1))
        return x
        

class ACNet(nn.Module):
    '''
    The actor-critic network for the PushEnv
    
    Attributes: 
        embd_dim:           the dimension of the embedded features for each input
        img_conv_list:      the architecture of the convolutional layers for image embedding 
        img_fc_list:        the architecture of the linear layers for image embedding 
        eef_pos_embd_list:  the architecture of the linear layers for end-effector pose embedding 
        policy_net_list:    the architecture of the policy net 
        value_net_list:     the architecture of the value net 
        test:               determines whether the object posisiton and goal position will be used for the network 
        obj_pos_embd_list:  the architecture of the linear layers for object posiion embedding 
        goal_pos_embd_list: the architecture of the linear layers for goal posiion embedding 
    '''
    def __init__(self, 
                embd_dim=16,
                img_conv_list=[4, 'P', 8, 'P', 16], 
                img_fc_list=[64, 16], 
                eef_pos_embd_list=[16],
                policy_net_list=[32, 2],
                value_net_list=[32, 1],
                test=False, 
                obj_pos_embd_list=[16], 
                goal_pos_embd_list=[16], 
                add_noise=True
                ):
        super().__init__()
        
        self.test = test
        self.add_noise = add_noise
        
        # design the feature extraction network, respectively 
        self.img_embd_net = ConvFCNet(IMAGE_SHAPE, img_conv_list, img_fc_list)
        self.touch_embd_net =  nn.Embedding(2, embd_dim)
        self.eef_pos_embd_net = create_fc_layers(EEF_POS_SHAPE, eef_pos_embd_list)
        
        if self.test:
            self.obj_pos_embd_net = create_fc_layers(OBJ_POS_SHAPE, obj_pos_embd_list)
            self.goal_pos_embd_net = create_fc_layers(GOAL_POS_SHAPE, goal_pos_embd_list)

        self.concat_size = embd_dim * 5 if self.test else embd_dim * 3 # determine size of concatenated features
        
        self.policy_net_mean = create_fc_layers(self.concat_size, policy_net_list, last_act=torch.nn.Tanh())
        self.policy_net_var = create_fc_layers(self.concat_size, policy_net_list, last_act=torch.nn.ReLU())
        self.value_net = create_fc_layers(self.concat_size, value_net_list)
        
    def forward(self, obs, target_v, reward):
        '''
        Args:
            obs: a list containing all the observations from the environment. 
                 with an order of [img, eef_pos, touch, obj_pos (optional), goal_pos (optional)]
        '''
        if self.test:
            img, eef_pos, touch, obj_pos, goal_pos = obs # unpack the observations
            obj_pos_embd = self.obj_pos_embd_net(obj_pos)
            goal_pos_embd = self.goal_pos_embd_net(goal_pos)
        else:
            img, eef_pos, touch = obs
        
        # extract features for each observation 
        img_embd = self.img_embd_net(img)
        eef_pos_embd = self.eef_pos_embd_net(eef_pos)
        touch_embd = self.touch_embd_net(touch).squeeze(1)
        
        # concatenate the extracted featurs
        if self.test:
            x = torch.cat([img_embd, eef_pos_embd, touch_embd, obj_pos_embd, goal_pos_embd], dim=1)
        else:
            x = torch.cat([img_embd, eef_pos_embd, touch_embd])
            
        policy_mean = self.policy_net_mean(x)
        value = self.value_net(x)
        
        if self.add_noise:
            policy_std = self.policy_net_var(x)
            noise = torch.rand_like(policy_std)
            policy = policy_mean + noise * policy_std       
        else:
            policy = policy_mean  
            policy_std = torch.zeros_like(policy_mean)   
        
         
        loss = self.calculate_loss(policy, policy_mean, policy_std, value, reward, target_v)
           
        return policy, value, loss
    
    def calculate_loss(self, policy, policy_mean, policy_std, value, reward, target_v):
        dist = torch.distributions.Normal(policy_mean[0], policy_std[0]+1e-5)
        policy_loss = - torch.sum(value * dist.log_prob(policy))
        value_loss = mse_fcn(target_v, value)
        entropy_loss = -0.5 - 0.5 * torch.log(2 * torch.tensor(np.pi) * torch.exp(torch.tensor(1)) \
                                              * (policy_std - 1 + 1e-5).pow(2)).sum()
        
        loss = policy_loss + value_loss + 0 * entropy_loss
        
        return loss
        



