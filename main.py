import torch

x = torch.randn(4, 8, 16)     # (B=4, T=8, C=16)


print(x)
print(x.mean(dim=(0,1)))


