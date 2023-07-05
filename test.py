import torch.nn as nn
import torch


def watch_weights(weights_dict):
    for name, weight in weights_dict.items():
        print(name, weight)


linear = nn.Linear(3, 3)
print(linear.weight)
# weights_test = linear.state_dict()
# print('org weights:')
# watch_weights(weights_test)
#
# a, b = torch.randn_like(linear.weight), torch.randn_like(linear.bias)
# print(f"randn create:{a}\n{b}")
# weights_dict = {'weight': a, 'bias': b}
#
# # linear.load_state_dict(weights_dict)
# # print(f'load_state_dict:')
# # watch_weights(linear.state_dict())
#
# linear.weight.data.copy_(weights_dict['weight'])
# print("after:")
# watch_weights(linear.state_dict())
#
# a = [1, 2]
# [b, c] = a
# print(b, c)
