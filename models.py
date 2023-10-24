import torch 
import torch.nn as nn
from functools import reduce

IMAGE_SHAPE = (3, 240, 240)
EEF_POS_SHAPE = 3
OBJ_POS_SHAPE = 3
GOAL_POS_SHAPE = 3


def create_conv_layers(in_channels, architecture):
    '''
    INPUT: 
    in_channels: Integer
    architecture: list
    
    OUTPUT: the corresponding pytorch sequential model
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


def create_fc_layers(in_features, architecture):
    '''
    INPUT: 
    infeatures: Integer
    architecture: list
    
    OUTPUT: the corresponding pytorch sequential model
    '''
    layers = []      
    for x in architecture:
        out_features = x
        # print(in_features, out_features)
        layers += [nn.Linear(in_features, out_features), nn.ReLU()]
        in_features = x

    layers.pop(-1)

    return nn.Sequential(*layers)


    
class ImageEmbd(nn.Module):
    def __init__(self, 
                 input_dim,
                 conv_layer_list, 
                 fc_layer_list):
        super().__init__()
        # encoder
        self.in_channels = input_dim[0]
        self.conv_layers = create_conv_layers(self.in_channels, conv_layer_list)
        
        sample_data = torch.zeros(1, self.in_channels, input_dim[1], input_dim[2])
        sample_out = self.conv_layers(sample_data)
        output_dim = reduce(lambda x, y: x * y, sample_out.shape)
        self.output_dim = output_dim
        
        self.fc_layers = create_fc_layers(output_dim, fc_layer_list)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.conv_layers(x)
        x = self.fc_layers(x.reshape(B, -1))
        
        return x
        

class ACNet(nn.Module):
    def __init__(self, 
                embd_dim=16,
                img_conv_list=[16, 'M', 32], 
                img_fc_list=[64, 16], 
                eef_pos_embd_list=[16],
                policy_net_list=[32, 2],
                value_net_list=[32, 1],
                test=False, 
                obj_pos_embd_list=[16], 
                goal_pos_embd_list=[16]
                ):
        super().__init__()
        
        self.test = test
        
        self.img_embd_net = ImageEmbd(IMAGE_SHAPE, img_conv_list, img_fc_list)
        self.touch_embd_net =  nn.Embedding(2, embd_dim)
        self.eef_pos_embd_net = create_fc_layers(EEF_POS_SHAPE, eef_pos_embd_list)
        
        if self.test:
            self.obj_pos_embd_net = create_fc_layers(OBJ_POS_SHAPE, obj_pos_embd_list)
            self.goal_pos_embd_net = create_fc_layers(GOAL_POS_SHAPE, goal_pos_embd_list)
            
        self.concat_size = embd_dim * 5 if self.test else embd_dim * 3 # size of concatenated features
        
        
        self.policy_net = create_fc_layers(self.concat_size, policy_net_list)
        self.value_net = create_fc_layers(self.concat_size, value_net_list)
        
    def forward(self, obs):
        if self.test:
            img, eef_pos, touch, obj_pos, goal_pos = obs
            obj_pos_embd = self.obj_pos_embd_net(obj_pos)
            goal_pos_embd = self.goal_pos_embd_net(goal_pos)
        else:
            img, eef_pos, touch = obs
            
        img_embd = self.img_embd_net(img)
        eef_pos_embd = self.eef_pos_embd_net(eef_pos)
        touch_embd = self.touch_embd_net(touch)
        
        
        x = torch.concat([img_embd, eef_pos_embd, touch_embd])
        if self.test:
            x = torch.concat([x, obj_pos_embd, goal_pos_embd])
            
        policy = self.policy_net(x)
        value = self.value(x)
        
        return policy, value
        
            
        
print("start")



x = torch.randn((1, 3)).unsqueeze(0)
print(x.shape)
print(eef_pos_embd(x).shape)

print("test finished")


