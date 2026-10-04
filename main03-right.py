import torch


pred   = torch.tensor([1.0, 2.0, 3.0, 4.0])      # (4,)   预测值
target = torch.tensor([[1.1], [2.1], [2.9], [4.2]])  # (4,1) 真实值，多了一维

target = target.squeeze(-1)      # (4,1) → (4,)

diff = pred - target
print(diff.shape)     # ← 你以为是 (4,) 或 (4,1)
print(diff)

loss = (diff ** 2).mean()
print(loss)           # ← 这个数字是错的，但程序不报错

