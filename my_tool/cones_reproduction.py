import torch
from collections import defaultdict
import json
from tqdm import tqdm
import os
# SavePath = r'D:\seekoo\SD\sd-scripts\cones\concept_neurons.json'


"""
一. 什么是Cones？
    目前简单的线性融合都尝试了，应该要去挖掘更深层次的融合方式，Cones的目标，找到指定的神经元Θ，对这些神经元缩小α倍可以使Loss变小
    对应的提出一个超参 ρ = (1-α)(Θ * ∂L/∂Θ)^-1，有：
    L(αΘ) = L(Θ(1-ρΘ * ∂L/∂Θ)) ≈ L(Θ) - ρΘ^2(∂L/∂Θ)^2   从而推出 ==>    Θ(∂L/∂Θ)>0
    即满足上述条件即为concept neurons

为什么需要Cones？
    如果Cones的理论是正确的，则表明神经网络中存在concept nerous对应一些主题信息，虽然极大概率表征的是隐式特征，
    但是对于某一簇concept nerous组合或许可以反映一些显性的特征，比如笔触，色彩......所以我们是不是应该探究一下如何定位concept neurons
    并统计一下不同风格之间concept neurons重合的比重有多高，以及重合和不重合的concept neurons分别有什么作用。

    a.是能否把重合的concept nerous进一步分解，探究每一小簇concept nerous对应什么特征
    b.探究一下不重合的concept nerous有什么作用，并且尝试进一步分解
    

Cones复现思路
作者通过提出和缩放参数α有一定联系的参数ρ, 从而推导出当Lcon减小时有 Θ * ∂L/∂Θ > 0, 通过K次采样若累积的Θ * ∂L/∂Θ > τ即表示该 neuron 为concept neuron
实现步骤：
1. 在计算loss并反向传播后不更新参数，先统计一下所有LoRA模块的参数并保存到字典，
    a.键名为"lora_name"，   e.g 'lora_te_text_model_encoder_layers_0_self_atten_k_proj'
    b.值为字典
        (a. 键名为"named_parameters()返回的参数名；   e.g：lora_up.weights"
        (b. 值为列表，每个元素为字典
            ((a. 键名为采样序号
            ((b. 值为字典, 两个键值对，分别存储参数和梯度
2. 根据Θ(k+1) = Θ(k) ⊙ (1-ρΘ(k) ⊙ ∂L/∂Θ(k) ) 进行密集采样，        ps：这一步是为了减少误差吗？
3. 重复K次1，2，最后输出包含K次采样的字典
4. 对于每个参数计算K次采样累积的 Θ(k) ⊙ ∂L/∂Θ(k) 判断累计值是否大于 τ，是则存入dict，最后保存到json


Schedule:
step 1：
✔   尝试在自定义网络（Conv和Linear）中计算梯度并验证正确性    目的：判断自己对梯度反传计算的理解是否正确
step 2：
✔   在LoRA计算完loss并反向传播后，遍历所有lora模块，计算Θ * ∂L/∂Θ，对于conv和linear应该计算 Θ(k) ⊙ ∂L/∂Θ(k)
step 3:
✔   统计所有符合条件的neurons，并分别存储到json文件
step 4：
    尝试基于Merge param代码上加入shut down concept neurons的代码，看看shut down 前后文生图效果差异
step 5：
    尝试各种超参影响，这里需要写一个concept neurons重合度匹配的脚本
        比如在不同epoch模型文件初始化条件下寻找concept neurons
        尝试不同的阈值选择对concept neurons影响
        尝试不用的rho选择对concept neurons影响
        舱室不同的采样次数对concept neurons影响
    
"""


class ConesState:
    def __init__(self):
        self.data = None
        self.grad = None
        self.Mp = None

    def update(self, data, grad):
        self.data = data
        self.grad = grad
        test = torch.mul(self.data, self.grad)
        # print(test.clone().flatten()[0])
        self.Mp = test if self.Mp is None else self.Mp + test


class Cones:
    def __init__(self, lora_network, train_text_encoder, train_unet, rho):
        def _get_models():
            cones_dict = defaultdict(lambda: defaultdict(ConesState))
            text_encoder_loras = lora_network.text_encoder_loras if train_text_encoder else []
            unet_loras = lora_network.unet_loras if train_unet else []

            lora_models = text_encoder_loras + unet_loras

            for lora in lora_models:
                for name, _ in lora.named_parameters():
                    cones_dict[lora.lora_name][name] = ConesState()

            return lora_models, cones_dict

        self.lora_models, self.cones_states = _get_models()
        self.neurons_dict = defaultdict(lambda: defaultdict(dict))
        self.rho = rho

    def save_parameter_state(self):
        for model in tqdm(self.lora_models):
            # neurons_param_dict = self.neurons_dict[model.lora_name]
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.cones_states[model.lora_name][name].update(param.data, param.grad)
                    # neurons_param_dict[name] = {
                    #         'sample_index': sample_index,
                    #         'param data': param.data.tolist(),
                    #         'grad': param.grad.tolist(),
                    #     }
            # for name, parameter in neurons_dict.items():
            #     print(name, parameter)
        # save_dir = r'D:\seekoo\SD\sd-scripts\cones'
        # save_path = os.path.join(save_dir, f'{sample_index}')
        # print(f"Saving {sample_index} sample concet neurons to >>{save_path}<< ......")
        # with open(save_path, 'w') as json_file:
        #     json.dump(self.neurons_dict, json_file)
        # print(f"Successfully wrote to >>{save_path}<< ")

        self.update_param()

    def update_param(self):
        for model in self.lora_models:
            model.requires_grad_(False)
            for name, param in model.named_parameters():
                mid_state = torch.mul(self.rho * param.data, param.grad)
                alpha = 1 - mid_state
                new_param = torch.mul(param.data, alpha)
                param.copy_(new_param)
            model.requires_grad_(True)

    # def _conver_to_json_specification(self, data_dict):
    #     data = defaultdict(lambda: defaultdict(list))
    #     for lora_name in data_dict.keys():
    #         neurons_param_dict = data_dict
    #
    #     return data

    def find_concept_neurons(self, save_dir, th):
        concept_neurons_dict = defaultdict(lambda: defaultdict(dict))
        general_neurons_dict =defaultdict(lambda: defaultdict(dict))
        for lora_name, param_dict in self.cones_states.items():
            for param_name, cone in param_dict.items():
                Mp_score = torch.sum(cone.Mp)
                if Mp_score <= th:
                    general_neurons_dict[lora_name][param_name] = {
                        'Mp':   Mp_score.tolist()
                    }
                else:
                    concept_neurons_dict[lora_name][param_name] = {
                        'Mp':   Mp_score.tolist()
                    }

        save_path = os.path.join(save_dir, 'concept_neurons.json')
        print(f"Saving concept neurons to >>{save_path}<< ......")
        with open(save_path, 'w')as json_file:
            json.dump(concept_neurons_dict, json_file)

        save_path = os.path.join(save_dir, 'general_neurons.json')
        print(f"Saving concept neurons to >>{save_path}<< ......")
        with open(save_path, 'w')as json_file:
            json.dump(general_neurons_dict, json_file)

        print(f"Success writing info to json ~~")

        # for lora_name in tqdm(self.neurons_dict.keys()):
        #     neurons_param_dict = self.neurons_dict[lora_name]
        #     for param_key in neurons_param_dict.keys():     # lora_down.weigts
        #         param_list = neurons_param_dict[param_key]
        #         Mp = None
        #         for param in param_list:
        #             if 'param data' not in param.keys():
        #                 continue
        #             data = torch.tensor(param['param data'])
        #             grad = torch.tensor(param['grad'])
        #             Mp = torch.mul(data, grad) if Mp is None else Mp + torch.mul(data, grad)
        #
        #         if Mp is not None and torch.sum(Mp) > th:
        #             concept_neurons_dict[lora_name][param_key]['Mp'] = Mp
        #             concept_neurons_dict[lora_name][param_key] += param_list
        #
        # print(f"Saving all concet neurons to >>{save_path}<< ......")
        # # data = self._conver_to_json_specification(concept_neurons_dict)
        # with open(save_path, 'w') as json_file:
        #     json.dump(concept_neurons_dict, json_file)
        # print(f"Successfully wrote to >>{save_path}<< ")

    def save_neurons(self, save_path):
        concept_neurons_dict = defaultdict(lambda: defaultdict(dict))
        for lora_name, param_dict in tqdm(self.cones_states.keys()):
            for param_name, cone in param_dict.item():
                Mp_score = torch.sum(cone.Mp)
            concept_neurons_dict[lora_name][param_name] = {
                'Mp': Mp_score
            }
        # data = self._conver_to_json_specification(self.neurons_dict)
        print(f"Saving all neurons to >>{save_path}<< ......")
        with open(save_path, 'w') as json_file:
            json.dump(self.neurons_dict, json_file)
        print(f"Successfully wrote to >>{save_path}<< ")


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import random
    import numpy as np

    seed = 1
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1)
            # self.conv.requires_grad_(False)
            self.linear = nn.Linear(1 * 2 * 2, 1)
            self.linear.requires_grad_(False)

            # with torch.no_grad():
                # self.conv.requires_grad_(False)
                # self.linear.requires_grad_(False)

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            x = self.linear(x)
            return x

    model = MyModel()

    input_data = torch.randn(1, 1, 4, 4)

    # labels = torch.randint(0, 10, (2,))
    target = torch.randn(1, dtype=torch.float)

    criterion = nn.functional.mse_loss
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    outputs = model(input_data)
    loss = criterion(outputs, target)

    optimizer.zero_grad()
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient for parameter {name}:")
            print(param.grad)

    # check_guidance()
