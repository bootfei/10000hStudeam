# 阶段 0：PyTorch 手感手册（v2）

> 目标：建立"张量思维"，能徒手写训练循环，看得懂形状**和数值**怎么流动。
> 时长：约 8 小时，建议分 4 次做完。
> **纪律：练习代码全程手打，不要用 AI 生成。**可以问概念、问报错，不能让它替你写。

---

## v2 改了什么

v1 只告诉你结果的形状，没告诉你**每个位置上的数是从哪来的**。这样只能防住"报错"类的 bug，防不住"不报错但算错"的 bug，而后者才是真正花时间的那一类。

v2 对每个操作都回答三个问题，你做练习时也按这三问来：

1. **形状**：结果是什么形状？
2. **来源**：结果的 `[i][j]` 位置，是由输入的哪些元素、怎么算出来的？
3. **手算**：拿一个 2×3 的小例子，在纸上算出每个数，再跑代码对一遍。

能把第 2 问写成一行式子（比如 `out[i] = x[i][0] + x[i][1] + x[i][2]`），这个操作才算真懂。

---

## 目录

- [0. 环境准备](#0-环境准备)
- [1. 心智模型：张量是什么](#1-心智模型张量是什么)
- [2. 形状与数值：reduce、索引、矩阵乘](#2-形状与数值reduce索引矩阵乘)
- [3. 广播：形状规则 + 复制规则](#3-广播形状规则--复制规则)
- [4. view / reshape / transpose / contiguous](#4-view--reshape--transpose--contiguous)
- [5. autograd 心智模型](#5-autograd-心智模型)
- [6. 手算梯度对拍](#6-手算梯度对拍)
- [7. 实战：徒手 MLP 训练 MNIST](#7-实战徒手-mlp-训练-mnist)
- [8. 常见报错速查](#8-常见报错速查)
- [9. 出师测验](#9-出师测验)

---

## 0. 环境准备

```bash
pip install torch torchvision matplotlib
```

```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")           # Apple Silicon
else:
    device = torch.device("cpu")

print(device, torch.__version__)
torch.manual_seed(1337)   # 从第一天就固定种子
```

阶段 0 全程用 CPU 就够。

---

## 1. 心智模型：张量是什么

一个 Tensor 由四样东西定义：

| 属性 | 含义 | 查看方式 |
|---|---|---|
| **shape** | 各维度大小 | `x.shape` |
| **dtype** | 元素类型 | `x.dtype` |
| **device** | 在 CPU 还是 GPU | `x.device` |
| **stride** | 每个维度走一步，在内存里跳几个元素 | `x.stride()` |

stride 在第 4 节展开。

### 底层是一条直线

不管张量有几维，数据在内存里都是**一条连续的直线**。多维只是"怎么解读这条直线"。

```python
x = torch.arange(6).view(2, 3)
# 内存：  0 1 2 3 4 5
# 解读成 2 行 3 列：
# [[0, 1, 2],
#  [3, 4, 5]]
```

规则是**按行填**：先填满第 0 行，再填第 1 行。这叫行优先（row-major）。第 4 节所有内容都建立在这一点上。

### 视图会共享内存

```python
a = torch.arange(6)
b = a[2:5]        # b 是 a 的一段视图，不是副本
b[0] = 999
print(a)          # tensor([  0,   1, 999,   3,   4,   5])  ← a 也变了
```

类似 Java 的 `ByteBuffer.slice()`。想要真正的副本，用 `.clone()`。

### 没有编译器兜底

Java 里形状不对编译期就报错。PyTorch 很多时候**不报错，只是算出错的数**（第 3 节会亲手制造一次）。检查习惯要自己建。

### 建张量

```python
torch.tensor([[1., 2.], [3., 4.]])   # 从 Python 数据
torch.zeros(2, 3)
torch.ones(2, 3)
torch.arange(6)                       # 0..5，注意是 int64
torch.randn(2, 3)                     # 标准正态，训练里最常用
torch.rand(2, 3)                      # [0,1) 均匀
```

**dtype 的坑**：`arange` 出来是 `int64`，整型张量不能求梯度。神经网络里除了 index 和 label，一律用 float：`torch.arange(6).float()`。

---

## 2. 形状与数值：reduce、索引、矩阵乘

本节用同一个小张量贯穿，**每个例子都先手算再运行**：

```python
x = torch.tensor([[3., 1., 2.],
                  [0., 5., 4.]])      # (2, 3)，记 x[i][j]，i 是行，j 是列
```

### 2.1 reduce：sum / mean / max

**`dim=k` 的意思：让第 k 个下标跑遍所有取值，把这些元素合并成一个。** 第 k 维因此消失。

```python
x.sum(dim=1)     # out[i] = x[i][0] + x[i][1] + x[i][2]
                 # = [3+1+2, 0+5+4] = [6., 9.]          形状 (2,)

x.sum(dim=0)     # out[j] = x[0][j] + x[1][j]
                 # = [3+0, 1+5, 2+4] = [3., 6., 6.]     形状 (3,)

x.mean(dim=1)    # out[i] = (x[i][0] + x[i][1] + x[i][2]) / 3
                 # = [2., 3.]                            形状 (2,)
```

一个好记的说法：`dim=1` 是"沿着列的方向扫过去"，每一行压成一个数；`dim=0` 是"沿着行的方向扫下来"，每一列压成一个数。

**keepdim**：不删掉那一维，而是把它留成大小 1。**数值完全一样，只是形状不同。**

```python
x.sum(dim=1, keepdim=True)   # [[6.],
                             #  [9.]]     形状 (2, 1)
```

为什么要留着这个 1？为了下一步能正确广播（3.4 节有个反例，不留就出错）。

**max 返回两样东西**：最大值，和最大值所在的位置。

```python
r = x.max(dim=1)
r.values     # [3., 5.]    每行最大值        形状 (2,)
r.indices    # [0, 1]      在该行的第几列    形状 (2,)，int64

x.argmax(dim=1)   # [0, 1]   只要位置时用这个
```

第 7 节判断"模型预测的是哪个数字"，用的就是 `argmax(dim=1)`。

**推广到三维**。对 `(B, T, C)`：

```python
x3.sum(dim=1)         # out[b][c] = Σ_t x3[b][t][c]    → (B, C)
                      # 含义：把每个样本的 T 个 token 向量加在一起
x3.mean(dim=-1)       # out[b][t] = 该 token 的 C 个特征取平均 → (B, T)
                      # 含义：LayerNorm 的第一步
```

### 2.2 索引：整数吃维，切片保维，None 加维

换一个更大的例子，数值就等于"它在内存里的位置"，方便看出取的是谁：

```python
y = torch.arange(24).view(2, 3, 4)
# y[0] = [[ 0,  1,  2,  3],      y[1] = [[12, 13, 14, 15],
#         [ 4,  5,  6,  7],              [16, 17, 18, 19],
#         [ 8,  9, 10, 11]]              [20, 21, 22, 23]]
```

**规则一：整数索引会吃掉那一维。**

```python
y[:, 1]        # 每个 y[b] 取第 1 行
               # [[ 4,  5,  6,  7],
               #  [16, 17, 18, 19]]           形状 (2, 4)  ← dim1 没了

y[:, -1, :]    # 每个样本取最后一行
               # [[ 8,  9, 10, 11],
               #  [20, 21, 22, 23]]           形状 (2, 4)
               # nanoGPT 生成文本时"只看最后一个 token"就是这一句

y[..., 0]      # ... 表示"前面所有维度都全取"，等价于 y[:, :, 0]
               # 每行取第 0 个元素
               # [[ 0,  4,  8],
               #  [12, 16, 20]]               形状 (2, 3)
```

**规则二：切片会保留那一维，哪怕只切出 1 个。**

```python
y[:, 1:2]      # [[[ 4,  5,  6,  7]],
               #  [[16, 17, 18, 19]]]         形状 (2, 1, 4)
```

数值和 `y[:, 1]` 一模一样，只是多了一层括号。

**规则三：`None` 在它所在的位置插入一个大小为 1 的新维。**

```python
v = torch.tensor([10, 20, 30])    # (3,)
v[None, :]     # [[10, 20, 30]]           形状 (1, 3)   一行
v[:, None]     # [[10], [20], [30]]       形状 (3, 1)   一列
```

`unsqueeze(k)` 等价于在第 k 个位置写 `None`；`squeeze(k)` 反过来，删掉大小为 1 的第 k 维。

**读多个索引时从左到右逐个处理**，没写到的维度原样补在后面：

```python
y[0, :, None]  # 0 吃掉 dim0 → 剩 (3, 4)
               # : 取 dim1（3）
               # None 在这里插入 1
               # dim2（4）没写到，补在后面
               # 形状 (3, 1, 4)
```

### 2.3 逐元素运算 vs 矩阵乘

这是两种完全不同的乘法，**形状相同时两个都可能不报错**，必须分清：

```python
a = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])          # (2, 3)

a * a      # 逐元素：out[i][j] = a[i][j] * a[i][j]
           # [[ 1.,  4.,  9.],
           #  [16., 25., 36.]]            形状 (2, 3)，和输入一样
```

**矩阵乘 `@`**：`out[i][j] = Σ_k a[i][k] * b[k][j]`，即 **a 的第 i 行和 b 的第 j 列做点积**。

```python
b = torch.tensor([[1., 0.],
                  [0., 1.],
                  [1., 1.]])              # (3, 2)

a @ b
# out[0][0] = 1*1 + 2*0 + 3*1 = 4
# out[0][1] = 1*0 + 2*1 + 3*1 = 5
# out[1][0] = 4*1 + 5*0 + 6*1 = 10
# out[1][1] = 4*0 + 5*1 + 6*1 = 11
# [[ 4.,  5.],
#  [10., 11.]]                            形状 (2, 2)
```

下标 k 在求和里被消掉了，所以**内维必须相等**（这里都是 3），结果形状是"外维拼起来"：`(2, 3) @ (3, 2) → (2, 2)`。

**高维的矩阵乘：只有最后两维做矩阵乘，前面的维度当成"批"，逐个配对。**

```python
x3 = torch.randn(4, 8, 16)      # (B, T, C)
W  = torch.randn(16, 32)        # (C, C_out)
(x3 @ W).shape                  # (4, 8, 32)
# out[b][t] = x3[b][t] @ W
# 含义：每个 token 的 16 维向量各自乘以 W，变成 32 维，token 之间互不影响
```

这就是"线性层只作用在最后一维"的数值含义。Transformer 里可以直接 `x @ W` 而不用管 B 和 T，原因就在这里。

```python
q = torch.randn(4, 8, 16)
k = torch.randn(4, 8, 16)
(q @ k.transpose(1, 2)).shape   # (4, 8, 16) @ (4, 16, 8) → (4, 8, 8)
# out[b][i][j] = q[b][i] · k[b][j]   （两个 16 维向量的点积）
# 含义：第 b 个样本里，第 i 个 token 和第 j 个 token 的相似度
# 这就是 attention score
```

### 练习 2.1：形状口算（先答再验）

```python
x = torch.randn(4, 8, 16)     # (B=4, T=8, C=16)

# 1. x.sum(dim=-1).shape
# 2. x.sum(dim=1).shape
# 3. x.sum(dim=1, keepdim=True).shape
# 4. x.mean(dim=(0,1)).shape
# 5. x.transpose(1,2).shape
# 6. (x @ torch.randn(16, 32)).shape
# 7. x.view(4, 8, 4, 4).shape
# 8. x.view(4, 8, 4, 4).transpose(1,2).shape
# 9. x[:, -1, :].shape
# 10. x[:, -1:, :].shape
```

<details>
<summary>答案</summary>

1. `(4, 8)`
2. `(4, 16)`
3. `(4, 1, 16)`
4. `(16,)`
5. `(4, 16, 8)`
6. `(4, 8, 32)`
7. `(4, 8, 4, 4)`
8. `(4, 4, 8, 4)`：多头注意力里 `(B,T,nh,hs) → (B,nh,T,hs)` 那一步
9. `(4, 16)`：整数吃维
10. `(4, 1, 16)`：切片保维
</details>

### 练习 2.2：数值手算（先答再验）

```python
x = torch.tensor([[3., 1., 2.],
                  [0., 5., 4.]])

# 1. x.max(dim=0).values 和 .indices 各是什么？
# 2. x.mean(dim=0) = ?
# 3. 写出 x.sum(dim=0) 的来源式子：out[j] = ?
# 4. x * x 和 x @ x.T 分别是什么？（x.T 是转置，形状 (3, 2)）
# 5. y = torch.arange(24).view(2, 3, 4)，y[1, :, -1] 的值和形状？
```

<details>
<summary>答案</summary>

1. values `[3., 5., 4.]`，indices `[0, 1, 1]`。每一列在两行里挑大的：第 0 列 3 比 0 大，在第 0 行；第 1 列 5 在第 1 行；第 2 列 4 在第 1 行。
2. `[1.5, 3., 3.]`
3. `out[j] = x[0][j] + x[1][j]`
4. `x * x = [[9., 1., 4.], [0., 25., 16.]]`，形状 (2, 3)。
   `x @ x.T`：out[i][j] 是第 i 行和第 j 行的点积。
   `[[3*3+1*1+2*2, 3*0+1*5+2*4], [0*3+5*1+4*2, 0*0+5*5+4*4]] = [[14., 13.], [13., 41.]]`，形状 (2, 2)。
5. `y[1]` 是第二块，每行取最后一个：`[15, 19, 23]`，形状 `(3,)`。
</details>

**通过标准**：两组都不看代码全对。错的题，把它的"来源式子"写出来再对一遍。

---

## 3. 广播：形状规则 + 复制规则

广播要同时懂两条规则：**形状规则**决定结果多大，**复制规则**决定每个位置是哪个数。只懂第一条，就会出现"形状对了、数全错"的情况。

### 3.1 形状规则

两个张量逐元素运算时，**从最后一维开始向前对齐**，每一维必须满足之一：

1. 两边相等
2. 有一边是 1
3. 有一边不存在（当作 1）

```
    (4, 8, 16)
       (8, 16)   →  (4, 8, 16)   ✓
    (4, 8, 16)
    (4, 1, 16)   →  (4, 8, 16)   ✓
    (4, 8, 16)
       (4, 16)   →  ✗ 8 和 4 对不上，报错
```

### 3.2 复制规则

> **大小为 1 或不存在的维度，会沿那个方向把同一份数据复制过去。**

**例一：行向量**

```python
A = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])      # (2, 3)
v = torch.tensor([10., 20., 30.])     # (3,)

# 形状：(2,3) 和 (3,) 右对齐，v 缺 dim0 → 当作 (1,3)
# 复制：v 被当成"一行"，往下复制成 2 行
#     [[10, 20, 30],
#      [10, 20, 30]]

A - v
# [[ -9., -18., -27.],
#  [ -6., -15., -24.]]
```

来源式子：`out[i][j] = A[i][j] - v[j]`。**第 j 列减的永远是 v[j]，和在哪一行无关。**

**例二：列向量**

```python
u = torch.tensor([[100.],
                  [200.]])            # (2, 1)

# 复制：u 的 dim1 是 1，往右复制成 3 列
#     [[100, 100, 100],
#      [200, 200, 200]]

A - u
# [[ -99.,  -98.,  -97.],
#  [-196., -195., -194.]]
```

来源式子：`out[i][j] = A[i][j] - u[i][0]`。**第 i 行减的永远是 u[i]。**

**记忆方法**：看那个张量"缺"或"为 1"的是哪一维，它就沿那一维被复制。行向量缺行，往下复制；列向量缺列，往右复制。

**例三：两边都被复制**

```python
p = torch.tensor([1., 2.])           # (2,)    → 当作 (1, 2)，往下复制
q = torch.tensor([[10.], [20.], [30.]])  # (3, 1) → 往右复制

p + q    # 形状 (3, 2)
# out[i][j] = p[j] + q[i]
# [[11., 12.],
#  [21., 22.],
#  [31., 32.]]
```

这一例最重要，下一节的静默 bug 就是它。

### 3.3 练习：制造一次静默错误

**务必亲手跑。**

```python
pred   = torch.tensor([1.0, 2.0, 3.0, 4.0])            # (4,)
target = torch.tensor([[1.1], [2.1], [2.9], [4.2]])    # (4,1)，多了一维

diff = pred - target
print(diff.shape)
print(diff)
loss = (diff ** 2).mean()
print(loss)
```

这和例三是同一个形状组合。按复制规则：**`diff[i][j] = pred[j] - target[i]`**，也就是每个预测值和每个真实值两两相减，得到 16 个差。

你本来想要的只有 4 个"一一对应"的差，它们恰好在对角线上（`i == j`）。其余 12 个都是无意义的数，全被算进了 loss。

**动手做三件事：**

1. 不跑代码，先填出 diff 的 4×4 表格，圈出对角线。
2. 手算正确的 loss：只取对角线的 4 个差，平方后取平均。
3. 跑代码，对比错误 loss 和正确 loss，算出差了几倍。

<details>
<summary>答案</summary>

```
            pred[0]=1  pred[1]=2  pred[2]=3  pred[3]=4
target[0]=1.1  -0.1       0.9        1.9        2.9
target[1]=2.1  -1.1      -0.1        0.9        1.9
target[2]=2.9  -1.9      -0.9        0.1        1.1
target[3]=4.2  -3.2      -2.2       -1.2       -0.2
```

正确 loss = (0.01 + 0.01 + 0.01 + 0.04) / 4 = **0.0175**
错误 loss = 16 个数平方的平均 = **2.5425**，大约是正确值的 145 倍。

程序一个错都不报。
</details>

正确写法，让两边形状一致：

```python
target = target.squeeze(-1)      # (4,1) → (4,)
```

### 3.4 实战中的同类 bug：忘写 keepdim

第 7 节的 softmax 要"每行减去该行的最大值"。用 3 个样本、3 个类别演示：

```python
logits = torch.tensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.]])     # (B=3, 类别=3)
```

**正确：保留维度**

```python
m = logits.max(dim=1, keepdim=True).values   # [[3.], [6.], [9.]]  (3, 1) 列向量
logits - m
# 复制规则：列向量往右复制，第 i 行减 m[i]，即减自己那一行的最大值 ✓
# [[-2., -1., 0.],
#  [-2., -1., 0.],
#  [-2., -1., 0.]]
```

**错误：忘了 keepdim**

```python
m = logits.max(dim=1).values                 # [3., 6., 9.]  (3,) 被当成行向量
logits - m
# 复制规则：行向量往下复制，第 j 列减 m[j]
# out[i][j] = logits[i][j] - (第 j 行的最大值)
# [[-2., -4., -6.],
#  [ 1., -1., -3.],
#  [ 4.,  2.,  0.]]
```

第 0 行本该是 `[-2, -1, 0]`，变成了 `[-2, -4, -6]`。这一行的 softmax 概率就被算错了，而且**不报错**。

如果 B 和类别数不相等，比如 `(64, 10) - (64,)`，最右一维 10 对 64，直接报错，反而安全。**只有尺寸恰好相等时，bug 才会静默。**

### 3.5 两条防御习惯

**一、关键位置写断言**，当单元测试用：

```python
assert pred.shape == target.shape, f"{pred.shape} vs {target.shape}"
```

**二、测试时每个维度用互不相同的尺寸。** 本手册一直用 `B=4, T=8, C=16` 就是这个原因。如果图省事用 `B=T=C=8`，维度用错了也能"恰好"跑通，bug 就藏起来了。

---

## 4. view / reshape / transpose / contiguous

### 4.1 view：按内存顺序重新切行

回到第 1 节：数据在内存里是一条直线。**view 不动数据，只换一种切法。**

```python
x = torch.arange(12).view(3, 4)
# 内存：0 1 2 3 4 5 6 7 8 9 10 11
# 每 4 个切一行：
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]

x.view(4, 3)
# 同一条直线，每 3 个切一行：
# [[ 0,  1,  2],
#  [ 3,  4,  5],
#  [ 6,  7,  8],
#  [ 9, 10, 11]]
```

### 4.2 transpose：行列互换

```python
x.transpose(0, 1)        # 也写作 x.T
# y[i][j] = x[j][i]
# [[ 0,  4,  8],
#  [ 1,  5,  9],
#  [ 2,  6, 10],
#  [ 3,  7, 11]]
```

**对比 4.1 和 4.2：`x.view(4, 3)` 和 `x.transpose(0, 1)` 形状都是 (4, 3)，但数值完全不同。** 只看形状，你分不出这两个操作；这就是 v1 的问题所在。

`permute` 是 transpose 的推广，一次性重排所有维度：

```python
z = torch.randn(2, 3, 4)
z.permute(2, 0, 1).shape    # (4, 2, 3)：新的 dim0 是原来的 dim2，依此类推
# out[a][b][c] = z[b][c][a]
```

### 4.3 stride：transpose 为什么不搬数据

stride 回答的是：**元素 `[i][j]` 在内存直线上的第几个位置？**

```python
x.stride()     # (4, 1)
# x[i][j] 的位置 = i*4 + j*1
# 例：x[2][1] 在位置 2*4+1 = 9，内存第 9 个就是 9 ✓
```

transpose 只是把 stride 也对调：

```python
y = x.transpose(0, 1)
y.stride()     # (1, 4)
# y[i][j] 的位置 = i*1 + j*4
# 例：y[1][2] 在位置 1+8 = 9 → 值是 9
# 对照 4.2 的表：y[1][2] 确实是 9 ✓
```

内存一个字节没动，只改了"怎么找元素"的公式。所以 transpose 几乎零成本。

### 4.4 contiguous：为什么 transpose 之后不能 view

**连续（contiguous）的意思：按行读这个张量的顺序，和它在内存里的顺序一致。**

- `x` 按行读：0 1 2 3 4 5 …，内存也是 0 1 2 3 4 5 …，连续 ✓
- `y = x.T` 按行读：0 4 8 1 5 9 2 6 10 …，内存是 0 1 2 3 …，**不连续**

`y.view(12)` 的要求是：把 y 按行读出的 12 个数原样排成一条。但 view 不能搬数据，只能换 stride；而 "0 4 8 1 5 9 …" 这种跳法没法用一个固定步长描述，所以报错：

```python
y.is_contiguous()          # False
y.view(12)                 # ✗ RuntimeError: view size is not compatible ...
y.contiguous().view(12)    # ✓ 先复制一份按行排好的新内存：0 4 8 1 5 9 2 6 10 3 7 11
y.reshape(12)              # ✓ reshape 能 view 就 view，不能就自动复制
```

| 操作 | 作用 | 搬内存吗 | 要求 |
|---|---|---|---|
| `view` | 换一种切法 | 永远不搬 | 必须 contiguous |
| `reshape` | 换一种切法 | 必要时复制 | 无 |
| `transpose(a,b)` | 交换两个维度 | 不搬，改 stride | 无 |
| `permute(...)` | 重排所有维度 | 不搬，改 stride | 无 |
| `contiguous()` | 按当前逻辑顺序复制一份 | 不连续时才搬 | 无 |

nanoGPT 习惯写 `.contiguous().view(...)` 而不是 `reshape`，是为了让"这里发生了一次复制"显式可见。两种写法结果一样。

### 4.5 多头切分：数值上发生了什么

用最小的例子：1 个样本、2 个 token、每个 token 4 维、分 2 个头（每头 2 维）。

```python
B, T, C, nh = 1, 2, 4, 2
hs = C // nh                         # 2

x = torch.arange(8).view(B, T, C).float()
# token0 = [0, 1, 2, 3]
# token1 = [4, 5, 6, 7]
```

**第一步：`view(B, T, nh, hs)`**，把每个 token 的 4 维切成 2 段：

```
token0 → [[0, 1],     ← 头 0 的部分
          [2, 3]]     ← 头 1 的部分
token1 → [[4, 5],
          [6, 7]]
```

**第二步：`transpose(1, 2)`**，变成 `(B, nh, T, hs)`，按头分组：

```
头 0 → [[0, 1],      ← token0 的头 0 部分
        [4, 5]]      ← token1 的头 0 部分
头 1 → [[2, 3],
        [6, 7]]
```

含义：**每个头拿到所有 token，但只拿各自那一段通道。** 之后每个头独立做 attention，互不干扰。

合并回来是反向操作：`transpose(1, 2)` 把头放回 token 内部，此时不连续，所以要 `contiguous().view(B, T, C)`。

### 练习 4.1：多头来回切换

看懂上面后**关掉手册**，自己写：

1. 用 `B, T, C, nh = 2, 5, 12, 3` 造一个随机 x
2. 切成 `(B, nh, T, hs)`，打印形状
3. 合并回 `(B, T, C)`，用 `torch.equal(x, z)` 验证得到 True
4. 再用 `torch.arange` 造一个 `(1, 2, 6)`、nh=3 的小张量，在纸上画出切分后每个头拿到哪些数，然后打印对照

### 练习 4.2：解释这个报错

```python
x = torch.randn(4, 8, 16)
w = x.view(4, 8, 4, 4).transpose(1, 2)    # (4, 4, 8, 4)
w.view(4, 4, 32)                          # ?
```

1. 先预测：会报错吗？
2. 打印 `x.view(4,8,4,4).stride()` 和 `w.stride()`，以及 `w.is_contiguous()`
3. 用自己的话解释原因，再给出两种修法

<details>
<summary>答案</summary>

会报错。`x.view(4,8,4,4)` 的 stride 是 `(128, 16, 4, 1)`；transpose 之后 `w` 的 stride 是 `(128, 4, 16, 1)`，不连续。

`view(4, 4, 32)` 要把最后两维 (8, 4) 合成一维 32。合并的前提是：按行读完一个大小为 4 的块后，下一个块紧挨着，也就是 dim2 的步长要等于 4×1 = 4。但这里 dim2 的步长是 16，中间隔着别的头的数据，无法用一个固定步长读出来。

修法：`w.contiguous().view(4, 4, 32)` 或 `w.reshape(4, 4, 32)`。
</details>

---

## 5. autograd 心智模型

### 5.1 梯度在数值上是什么

`W.grad[i][j]` 回答一个问题：**W[i][j] 增加一点点，loss 会变化多少倍？**

- 正数：W[i][j] 变大，loss 变大，所以应该把它调小
- 负数：反过来
- 绝对值大：这个参数对 loss 影响大

这就是更新式 `W -= lr * W.grad` 的含义：每个参数朝着让 loss 下降的方向挪一小步。`W.grad` 和 W 形状永远相同，因为每个参数各有一个梯度。

### 5.2 怎么算出来的

PyTorch 在前向运算时偷偷**录一盘磁带**（计算图），记下每一步操作。`loss.backward()` 倒着放这盘磁带，用链式法则把梯度算回去。

```python
x = torch.tensor([2.0], requires_grad=True)   # 叶子节点
y = x ** 2                                     # y 记住了"我是 x 平方来的"
z = y * 3

z.backward()
print(x.grad)     # tensor([12.])
# 手算：z = 3x²，dz/dx = 6x = 12 ✓
print(y.grad)     # None + 警告：非叶子节点默认不存梯度
```

三个要点：

**只有 `requires_grad=True` 的叶子张量才存 `.grad`。** 中间结果默认不存，省内存。

**`.grad` 是累加的，不是覆盖的：**

```python
x = torch.tensor([2.0], requires_grad=True)
(x**2).backward()
print(x.grad)          # tensor([4.])    d(x²)/dx = 2x = 4
(x**2).backward()
print(x.grad)          # tensor([8.])    4 + 4，累加了
```

所以训练循环每一步都要清零。忘了清零，每步用的是历史梯度之和，步子越来越大：loss 先降，然后震荡或爆炸。

**`backward()` 只能从标量出发。** loss 必须是一个数，所以最后总要 `.mean()` 或 `.sum()`。

### 5.3 三个"关掉磁带"的场景

```python
with torch.no_grad():          # 1. 推理/验证，不需要梯度
    logits = model(x)

with torch.no_grad():          # 2. 手动更新参数，更新本身不能被录进图
    W -= lr * W.grad

y = x.detach()                 # 3. 把张量从图里摘出来，共享内存但断开梯度
```

### 5.4 ⚠️ 手动更新参数的陷阱：必须原地改

```python
# ✗ 错误
with torch.no_grad():
    W1 = W1 - lr * W1.grad
```

`W1 - lr * W1.grad` 会**新建一个张量**，然后把名字 `W1` 指向它。因为是在 `no_grad` 里算的，这个新张量的 `requires_grad=False`。下一轮前向用它算 loss，梯度就不会流到它身上，`W1.grad` 是 None。

更隐蔽的写法：

```python
# ✗ 更隐蔽：不报错，但参数根本没更新
with torch.no_grad():
    for p in params:
        p = p - lr * p.grad    # 只改了局部变量 p，params 列表里的张量原封不动
```

这种写法 loss 纹丝不动，程序也不报错。

```python
# ✓ 正确：原地修改，张量还是原来那个
with torch.no_grad():
    for p in params:
        p -= lr * p.grad       # 等价于 p.sub_(lr * p.grad)
```

带下划线的方法（`sub_`、`add_`、`zero_`）都是**原地操作**，这是 PyTorch 的命名约定。Java 视角：前者是给引用重新赋值，后者是修改引用指向的对象。

### 5.5 loss 累加要用 .item()

```python
total += loss          # ✗ total 一直引用着整张计算图，内存越吃越多
total += loss.item()   # ✓ 变成 Python float，脱离计算图
```

---

## 6. 手算梯度对拍

### 练习 6.1：标量

```
f(x, y) = x² · y + y³        在 x = 2, y = 3 处：

∂f/∂x = 2xy       = 2·2·3       = 12
∂f/∂y = x² + 3y²  = 4 + 27      = 31
```

（`∂f/∂x` 读作"f 对 x 的偏导"：只让 x 变、y 当常数，f 变化的速率。）

```python
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
f = x**2 * y + y**3
f.backward()
print(x.grad, y.grad)      # 12.0 和 31.0
```

### 练习 6.2：矩阵乘法的梯度（逐元素推导）

设 X 是 (3, 4)，W 是 (4, 5)，`L = (X @ W).sum()`。

**第一步：把 L 写成元素式子。** `(XW)[i][j] = Σ_k X[i][k] · W[k][j]`，再对所有 i、j 求和：

```
L = Σ_i Σ_j Σ_k  X[i][k] · W[k][j]

（Σ_i 读作"让 i 跑遍所有取值，把后面的项全加起来"，
  三个 Σ 就是三层嵌套 for 循环里的累加）
```

**第二步：对某一个 W[k][j] 求偏导。** 只有含 W[k][j] 的项留下（对每个 i 各有一项）：

```
∂L / ∂W[k][j]  =  Σ_i X[i][k]  =  X 第 k 列之和
```

结果和 j 无关：梯度矩阵的每一行 k 都是同一个数复制 5 次。

**第三步：写成矩阵形式。** `X.T @ ones(3, 5)` 的 `[k][j]` 元素正好是 `Σ_i X[i][k] × 1`。

```python
X = torch.randn(3, 4)
W = torch.randn(4, 5, requires_grad=True)
L = (X @ W).sum()
L.backward()

manual = X.T @ torch.ones(3, 5)
print(torch.allclose(W.grad, manual))     # True
print(W.grad[:, 0], X.sum(dim=0))         # 两者应该相同
```

**形状检查**：`W.grad` 形状永远等于 `W`。阶段 2 里哪个梯度形状不对，说明前向写错了。

### 练习 6.3：数值梯度对拍

导数定义（ε 是一个很小的数，比如 0.0001）：

```
f'(x) ≈ ( f(x + ε) − f(x − ε) ) / (2ε)
```

把每个元素挨个推一点点，看 f 变多少：

```python
def numeric_grad(f, x, eps=1e-4):
    grad = torch.zeros_like(x)
    flat_x, flat_g = x.view(-1), grad.view(-1)
    for i in range(flat_x.numel()):
        orig = flat_x[i].item()
        flat_x[i] = orig + eps
        fp = f(x).item()
        flat_x[i] = orig - eps
        fm = f(x).item()
        flat_x[i] = orig
        flat_g[i] = (fp - fm) / (2 * eps)
    return grad

x = torch.randn(3, 4, dtype=torch.float64)   # float64，精度才够
func = lambda t: (t ** 3).sum()

x.requires_grad_(True)
func(x).backward()
print(torch.allclose(x.grad, numeric_grad(func, x.detach()), atol=1e-6))  # True
```

先自己推一下：`(t**3).sum()` 对 `t[i][j]` 的偏导是什么？（答：`3 * t[i][j]**2`）然后可以再加一行 `torch.allclose(x.grad, 3 * x**2)` 验证。

---

## 7. 实战：徒手 MLP 训练 MNIST

**规则：不许用 `nn.Linear`、`nn.Module`、`torch.optim`。**

### 7.1 准备数据

```python
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

train_set = datasets.MNIST(root="./data", train=True,  download=True,
                           transform=transforms.ToTensor())
test_set  = datasets.MNIST(root="./data", train=False, download=True,
                           transform=transforms.ToTensor())

Xtr = train_set.data.float().view(-1, 784) / 255.0    # (60000, 784)
Ytr = train_set.targets                                # (60000,) int64
Xte = test_set.data.float().view(-1, 784) / 255.0
Yte = test_set.targets
```

数值上发生了什么：
- `train_set.data` 是 `(60000, 28, 28)` 的 uint8，每个像素是 0–255 的灰度值
- `.view(-1, 784)`：每张图的 28×28 按行展开成 784 个数（第 1 节的行优先），`-1` 表示"这一维自动算"
- `/ 255.0`：缩放到 0–1
- `Ytr[n]` 是第 n 张图的真实数字，0–9

先看一眼数据：

```python
import matplotlib.pyplot as plt
plt.imshow(Xtr[0].view(28, 28), cmap="gray")
plt.title(f"label = {Ytr[0].item()}")
plt.show()
```

### 7.2 初始化参数

```python
torch.manual_seed(1337)
n_in, n_hidden, n_out = 784, 128, 10

W1 = (torch.randn(n_in, n_hidden) / n_in**0.5).requires_grad_()
b1 = torch.zeros(n_hidden, requires_grad=True)
W2 = (torch.randn(n_hidden, n_out) / n_hidden**0.5).requires_grad_()
b2 = torch.zeros(n_out, requires_grad=True)

params = [W1, b1, W2, b2]
print(sum(p.numel() for p in params), "parameters")
```

**为什么除以 `sqrt(n_in)`**：`(x @ W)[b][h] = Σ_k x[b][k] * W[k][h]`，是 784 个乘积之和。每个乘积的方差约为 1，784 个独立项相加，方差约 784，标准差约 √784 = 28。把 W 除以 28，输出标准差就回到 1 左右。不缩放的话，每过一层数值放大几十倍，几层之后就爆了。

```python
x = torch.randn(1000, 784)
print((x @ torch.randn(784, 128)).std())              # ≈ 28
print((x @ (torch.randn(784, 128)/784**0.5)).std())   # ≈ 1
```

### 7.3 前向：每一步的形状和含义

```python
def forward(X):
    h = X @ W1 + b1          # (B, 784) @ (784, 128) + (128,) → (B, 128)
    h = torch.relu(h)        # 逐元素：负数变 0，正数不变
    logits = h @ W2 + b2     # (B, 128) @ (128, 10) + (10,) → (B, 10)
    return logits
```

- `+ b1`：b1 是 `(128,)`，按 3.2 的复制规则往下复制 B 行，即**每个样本加同一个偏置**
- `logits[b][c]`：模型认为"第 b 张图是数字 c"的打分，越大越像。它还不是概率

### 7.4 手写交叉熵：用 3 个数走一遍

```python
def my_cross_entropy(logits, targets):
    logits = logits - logits.max(dim=1, keepdim=True).values      # ①
    log_probs = logits - logits.exp().sum(dim=1, keepdim=True).log()  # ②
    picked = log_probs[torch.arange(len(targets)), targets]       # ③
    return -picked.mean()                                          # ④
```

用一个样本、3 个类别、`logits = [[2., 1., 0.]]`、`target = [0]` 手算：

| 步骤 | 计算 | 结果 |
|---|---|---|
| ① 减最大值 | `[2,1,0] - 2` | `[0, -1, -2]` |
| exp | `[e⁰, e⁻¹, e⁻²]` | `[1, 0.3679, 0.1353]` |
| 求和再 log | `log(1.5032)` | `0.4076` |
| ② log 概率 | `[0,-1,-2] - 0.4076` | `[-0.4076, -1.4076, -2.4076]` |
| ③ 取正确类别 | target 是 0，取第 0 个 | `-0.4076` |
| ④ 取负、平均 | | **loss = 0.4076** |

验算：类别 0 的概率是 `1 / 1.5032 = 0.665`，`-ln(0.665) = 0.4076` ✓

**含义**：交叉熵 = −log(模型给正确答案的概率)。概率为 1 时 loss 为 0；概率越小，loss 越大。

**① 为什么减最大值**：`torch.tensor([1000.0]).exp()` 是 inf。每行减同一个常数，softmax 的结果不变（分子分母同乘 e^(-m)），但数值不再溢出。**必须是每行减自己的最大值**，所以要 keepdim（见 3.4）。

**③ 高级索引**：两个等长的索引张量逐个配对取值。

```python
# log_probs 形状 (3, 3)，targets = [2, 0, 1]
log_probs[torch.arange(3), targets]
# = [log_probs[0][2], log_probs[1][0], log_probs[2][1]]    形状 (3,)
# 即每个样本取出它正确类别的那一格
```

和官方实现对拍：

```python
logits = forward(Xtr[:32])
print(my_cross_entropy(logits, Ytr[:32]))
print(F.cross_entropy(logits, Ytr[:32]))     # 两个数必须一致
```

### 7.5 开跑前的两个检查

**检查一：初始 loss ≈ ln(类别数)**

没训练时，模型对 10 个类别应该差不多一样没把握，每类概率约 1/10，loss = −ln(1/10) = ln(10) ≈ 2.30。

```python
with torch.no_grad():
    init_loss = my_cross_entropy(forward(Xtr[:1000]), Ytr[:1000])
print(init_loss.item(), "expect ≈", torch.tensor(10.0).log().item())
```

如果是 8 或者 0.5，说明初始化或 loss 写错了，**往下训练全是浪费时间**。

**检查二：单 batch 过拟合到接近 0**

```python
Xb, Yb = Xtr[:32], Ytr[:32]
for i in range(300):
    loss = my_cross_entropy(forward(Xb), Yb)
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    loss.backward()
    with torch.no_grad():
        for p in params:
            p -= 0.1 * p.grad
    if i % 50 == 0:
        print(i, loss.item())
```

32 个样本对 10 万参数来说完全可以背下来，loss 必须掉到 0.01 以下。**降不下去，代码一定有 bug。** 这是全流程性价比最高的测试。

### 7.6 完整训练循环

```python
batch_size, lr, steps = 64, 0.1, 3000

for step in range(steps):
    idx = torch.randint(0, len(Xtr), (batch_size,))   # 随机抽 64 个下标
    Xb, Yb = Xtr[idx], Ytr[idx]                        # 高级索引取出这 64 行

    loss = my_cross_entropy(forward(Xb), Yb)

    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    loss.backward()

    with torch.no_grad():
        for p in params:
            p -= lr * p.grad

    if step % 300 == 0:
        print(f"step {step:5d}  loss {loss.item():.4f}")

with torch.no_grad():
    acc = (forward(Xte).argmax(dim=1) == Yte).float().mean()
print(f"test accuracy: {acc.item():.4f}")
```

**准确率这一行的数值过程**，假设只有 3 个样本：

```
forward(Xte).argmax(dim=1)  → [3, 1, 4]            每行打分最高的类别
Yte                         → [3, 1, 5]
==                          → [True, True, False]  逐元素比较
.float()                    → [1., 1., 0.]
.mean()                     → 0.667
```

正常应该在 0.96 以上；低于 0.90 就回去查形状。

### 7.7 四个破坏性实验

**故意搞坏它，观察现象。**每个都跑一遍，把 loss 曲线的样子记下来。

| 实验 | 改动 | 预期现象 | 原因 |
|---|---|---|---|
| 忘记清梯度 | 注释掉 `p.grad.zero_()` | loss 震荡或爆炸 | 梯度累加，步子越来越大（5.2） |
| 学习率过大 | `lr = 10.0` | loss 变 nan | 一步跨过头，数值溢出；nan 不可逆 |
| 学习率过小 | `lr = 0.0001` | loss 几乎不动 | 步子太小；现象和"有 bug"很像，要能区分 |
| 不缩放初始化 | 去掉 `/ n_in**0.5` | 初始 loss 远大于 2.30 | logits 数值很大，softmax 过于自信地猜错（7.2） |

### 7.8 收尾：换成官方 API

```python
import torch.nn as nn

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
opt = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(3000):
    idx = torch.randint(0, len(Xtr), (64,))
    loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
    opt.zero_grad()
    loss.backward()
    opt.step()
```

你应该能逐行说出这四行封装掉了你手写的哪部分。

---

## 8. 常见报错速查

| 报错 | 真实原因 | 处理 |
|---|---|---|
| `mat1 and mat2 shapes cannot be multiplied (AxB and CxD)` | 矩阵乘内维不匹配 | B 必须等于 C |
| `view size is not compatible with input tensor's size and stride` | 对不连续张量用 view | `.contiguous()` 或 `reshape`（4.4） |
| `element 0 of tensors does not require grad` | 参数没设 `requires_grad`，或参数被重新赋值（5.4） | 检查创建和更新方式 |
| `Trying to backward through the graph a second time` | 同一个 loss 调了两次 backward | 每步重新前向 |
| `Expected all tensors to be on the same device` | 一部分在 CPU 一部分在 GPU | 统一 `.to(device)` |
| `expected scalar type Long but found Float` | label 必须是 int64 | `targets.long()` |
| `grad can be implicitly created only for scalar outputs` | 对非标量调 backward | loss 加 `.mean()` |
| **不报错但 loss 不降** | 广播错了（3.3、3.4）、忘清梯度、参数没原地更新（5.4） | 单 batch 过拟合测试 |

**PyTorch 不报错 ≠ 代码正确。**

---

## 9. 出师测验

不查文档、不跑代码，全对才算过关。

### 形状题

```python
x = torch.randn(8, 16, 64)      # (B, T, C)
```

1. `x.mean(dim=-1, keepdim=True).shape`
2. `(x - x.mean(dim=-1, keepdim=True)).shape`
3. `x.view(8, 16, 8, 8).permute(0, 2, 1, 3).shape`
4. `(x @ torch.randn(64, 192)).split(64, dim=-1)` 得到几个张量，各是什么形状
5. `x[:, :4].shape` 和 `x[:, 4].shape`
6. `x.transpose(1,2) @ x` 的形状

### 数值题

```python
a = torch.tensor([[3., 1., 2.],
                  [0., 5., 4.]])
```

7. `a - a.mean(dim=1, keepdim=True)` 的值是什么？
8. `a.view(3, 2)` 和 `a.T` 的值分别是什么？
9. `torch.tensor([1., 2.]) + torch.tensor([[10.], [20.], [30.]])` 的值和形状？写出来源式子 `out[i][j] = ?`
10. `logits = [[0., 0., 0.]]`、`target = [2]`，交叉熵是多少？和"初始 loss"检查有什么关系？
11. `a.argmax(dim=0)` 的值？

### 概念题

12. `view` 和 `reshape` 的区别？什么时候必须 `contiguous()`？
13. 为什么训练循环必须 `zero_grad()`？不清零会怎样？
14. `W = W - lr * W.grad` 和 `W -= lr * W.grad` 的本质区别？
15. softmax 前减最大值，为什么必须 `keepdim=True`？不写会怎样，什么情况下不报错？
16. 单 batch 过拟合测试是干什么用的？降不到 0 说明什么？

<details>
<summary>答案</summary>

1. `(8, 16, 1)`
2. `(8, 16, 64)`：LayerNorm 的第一步
3. `(8, 8, 16, 8)`：多头切分
4. 3 个，各 `(8, 16, 64)`：nanoGPT 里一次算出 Q、K、V 的写法
5. `(8, 4, 64)` 和 `(8, 64)`
6. `(8, 64, 64)`
7. 每行均值是 `[[2.], [3.]]`，每行减自己的均值：`[[1., -1., 0.], [-3., 2., 1.]]`
8. `a.view(3, 2) = [[3., 1.], [2., 0.], [5., 4.]]`（内存顺序每 2 个切一行）；`a.T = [[3., 0.], [1., 5.], [2., 4.]]`（行列互换）。形状相同，值不同。
9. `[[11., 12.], [21., 22.], [31., 32.]]`，形状 (3, 2)，`out[i][j] = p[j] + q[i]`
10. 三个类别概率都是 1/3，loss = ln(3) ≈ 1.0986。和 target 是哪个无关。这正是"初始 loss ≈ ln(类别数)"的来源：未训练的模型打分接近相等。
11. `[0, 1, 1]`
12. 见 4.4
13. 见 5.2
14. 见 5.4
15. 见 3.4：不写时最大值是 `(B,)`，被当成行向量往下复制，第 j 列减的是第 j 行的最大值。B 恰好等于类别数时不报错，静默算错。
16. 见 7.5
</details>

### 动手题

17. 不看本手册，从空文件开始，20 分钟内写出 7.6 的完整训练循环，跑到测试准确率 96% 以上。

**第 17 题做到，阶段 0 结束，进阶段 1。**

---

## 附：和阶段 2 的对照

| 阶段 0 的知识点 | 阶段 2 的用处 |
|---|---|
| `(B,T,C)` 形状 + 来源式子 | 全程 |
| `x @ W` 逐 token 独立 | 所有线性层 |
| `q @ k.T` 的数值含义 | attention score |
| `view + transpose` | 多头切分（4.5） |
| `contiguous()` | 多头合并 |
| 广播复制规则 | mask 加到 attention score 上 |
| keepdim + 减最大值 | attention 内部的 softmax |
| 高级索引 | 交叉熵取 target 位置 |
| 缩放初始化 | GPT-2 的 0.02 初始化和残差缩放 |
| 初始 loss = ln(vocab) | 关卡一 |
| 单 batch 过拟合 | 关卡二 |
| 互不相同的测试尺寸 | 所有单元测试 |

---

**卡住超过一小时，先回到三招：打印形状、用小数字手算来源、单 batch 过拟合。**
