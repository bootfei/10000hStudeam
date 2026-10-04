// 预设游戏、章节和题库。每题是 [题目, 参考答案]。
// 题目来自 阶段0-PyTorch手感手册-v2.md 和 项目背景-AI训练工程师转型.md。
window.SEED = { games: [
  {
    name: 'PyTorch 手感（阶段 0）', emoji: '🔥', color: '#c0392b',
    chapters: [
      { name: '0. 环境准备', questions: [
        ['为什么从第一天就要 torch.manual_seed(...)？', '让结果可复现。对拍和排 bug 时，才能分清差异是代码改动造成的还是随机性造成的。'],
        ['在 Apple Silicon 上，device 的选择顺序是？', 'cuda → mps → cpu。阶段 0 全程用 CPU 就够。'],
      ]},
      { name: '1. 心智模型：张量是什么', questions: [
        ['一个 Tensor 由哪四样东西定义？', 'shape、dtype、device、stride。'],
        ['a = torch.arange(6); b = a[2:5]; b[0] = 999\na 现在是什么？为什么？', '[0, 1, 999, 3, 4, 5]。切片是视图，和 a 共享内存。要副本用 .clone()。'],
        ['torch.arange(6) 的 dtype 是什么？能直接求梯度吗？', 'int64，不能求梯度。除了 index 和 label，一律 .float()。'],
        ['torch.arange(6).view(2, 3) 的值是什么？“行优先”指什么？', '[[0,1,2],[3,4,5]]。内存是一条直线，先填满第 0 行再填第 1 行。'],
      ]},
      { name: '2. 形状与数值：reduce、索引、矩阵乘', questions: [
        ['x = torch.randn(8, 16, 64)\nx.mean(dim=-1, keepdim=True).shape = ?', '(8, 16, 1)'],
        ['x = torch.randn(8, 16, 64)\nx[:, :4].shape 和 x[:, 4].shape 分别是？', '(8, 4, 64) 和 (8, 64)。切片保维，整数吃维。'],
        ['x = torch.randn(8, 16, 64)\n(x.transpose(1, 2) @ x).shape = ?', '(8, 64, 64)'],
        ['x = torch.randn(8, 16, 64)\n(x @ torch.randn(64, 192)).split(64, dim=-1) 得到几个张量？各是什么形状？', '3 个，各 (8, 16, 64)。nanoGPT 一次算出 Q、K、V 的写法。'],
        ['a = [[3,1,2],[0,5,4]]\na.argmax(dim=0) = ?', '[0, 1, 1]'],
        ['x3 形状 (B, T, 16)，W 形状 (16, 32)。x3 @ W 的形状？out[b][t] 的含义？', '(B, T, 32)。out[b][t] = x3[b][t] @ W，每个 token 的向量各自乘 W，token 之间互不影响。'],
        ['q, k 形状都是 (B, T, 16)。(q @ k.transpose(1,2))[b][i][j] 的含义？', 'q[b][i] · k[b][j]：第 b 个样本里第 i 个 token 和第 j 个 token 的相似度，也就是 attention score。形状 (B, T, T)。'],
      ]},
      { name: '3. 广播：形状规则 + 复制规则', questions: [
        ['广播的形状规则是什么？', '右对齐；每一维要么相等，要么有一个是 1（缺失的维当作 1）。'],
        ['torch.tensor([1., 2.]) + torch.tensor([[10.], [20.], [30.]])\n值、形状、来源式子？', '[[11,12],[21,22],[31,32]]，形状 (3, 2)，out[i][j] = p[j] + q[i]。'],
        ['softmax 前减最大值，为什么必须 keepdim=True？不写会怎样？什么情况下不报错？', '不写时最大值形状是 (B,)，被当成行向量往下复制，第 j 列减的是第 j 行的最大值。B 恰好等于类别数时不报错，静默算错。'],
      ]},
      { name: '4. view / reshape / transpose / contiguous', questions: [
        ['a = [[3,1,2],[0,5,4]]\na.view(3, 2) 和 a.T 的值分别是？', 'view：[[3,1],[2,0],[5,4]]（按内存顺序每 2 个切一行）；a.T：[[3,0],[1,5],[2,4]]（行列互换）。形状相同，值不同。'],
        ['view 和 reshape 的区别？什么时候必须 contiguous()？', 'view 永远不搬内存，要求张量 contiguous；reshape 能 view 就 view，不能就自动复制。transpose/permute 之后不连续，view 前要 .contiguous()。'],
        ['x = torch.randn(8, 16, 64)\nx.view(8, 16, 8, 8).permute(0, 2, 1, 3).shape = ?', '(8, 8, 16, 8)。多头切分：(B, nh, T, hs)。'],
        ['x 形状 (3, 4)，stride 是 (4, 1)。x.T 的 stride 是什么？为什么 transpose 不搬数据？', '(1, 4)。transpose 只交换 stride，内存原封不动。'],
        ['nanoGPT 为什么写 .contiguous().view(...) 而不是 reshape？', '让“这里发生了一次复制”显式可见。结果一样。'],
      ]},
      { name: '5. autograd 心智模型', questions: [
        ['W.grad[i][j] 在数值上是什么意思？', 'W[i][j] 增加一点点，loss 会变化多少倍。正数说明该调小，绝对值大说明影响大。W.grad 和 W 形状相同。'],
        ['x = 2（requires_grad），y = x**2，z = 3*y，z.backward()。\nx.grad = ? y.grad = ?', 'x.grad = 12（dz/dx = 6x）。y.grad 是 None：非叶子节点默认不存梯度。'],
        ['.grad 是累加还是覆盖？忘记清零会出现什么现象？', '累加。每步用的是历史梯度之和，步子越来越大：loss 先降，然后震荡或爆炸。'],
        ['with torch.no_grad():\n    W1 = W1 - lr * W1.grad\n有什么问题？', '新建了一个 requires_grad=False 的张量，把名字 W1 指过去。下一轮梯度流不到它，W1.grad 是 None。要原地改：W1 -= lr * W1.grad。'],
        ['for p in params:\n    p = p - lr * p.grad\n会发生什么？', '只改了局部变量 p，params 里的张量原封不动。不报错，loss 纹丝不动。'],
        ['total += loss 有什么问题？', 'total 一直引用整张计算图，内存越吃越多。应写 total += loss.item()。'],
        ['“关掉磁带”的三个场景？', '推理/验证用 no_grad；手动更新参数用 no_grad；detach() 把张量从图里摘出来。'],
      ]},
      { name: '6. 手算梯度对拍', questions: [
        ['f(x, y) = x²y + y³，在 x=2, y=3 处，∂f/∂x 和 ∂f/∂y？', '∂f/∂x = 2xy = 12；∂f/∂y = x² + 3y² = 31。'],
        ['Y = X @ W，已知 dL/dY，dL/dW = ?', 'X.T @ dL/dY，形状和 W 相同。'],
        ['数值梯度对拍怎么做？', '中心差分 (f(x+h) − f(x−h)) / 2h，用 float64，和 autograd 的结果比相对误差。'],
      ]},
      { name: '7. 实战：徒手 MLP 训练 MNIST', questions: [
        ['10 分类模型的初始 loss 应该约等于多少？不对说明什么？', 'ln(10) ≈ 2.30。差太多说明初始化或 loss 写错了，往下训练全是浪费。'],
        ['logits = [[0,0,0]]，target = [2]，交叉熵是多少？', 'ln(3) ≈ 1.0986，和 target 是哪个无关。这就是“初始 loss ≈ ln(类别数)”的来源。'],
        ['单 batch 过拟合测试是干什么的？降不到 0 说明什么？', '32 个样本对十万参数完全可以背下来，loss 应掉到 0.01 以下。降不下去，代码一定有 bug。性价比最高的测试。'],
        ['手写交叉熵时，log_probs[range(B), targets] 在做什么？', '高级索引：每个样本取出它正确类别的那一格，得到形状 (B,)。'],
        ['lr = 10.0 和去掉初始化缩放 / n_in**0.5，各会出现什么现象？', 'lr 过大：loss 变 nan，不可逆。不缩放：初始 loss 远大于 2.30，softmax 过于自信地猜错。'],
        ['opt.zero_grad() / loss.backward() / opt.step() 分别替代了你手写的哪部分？', 'zero_grad = 每个 p.grad.zero_()；backward 不变；step = no_grad 下原地 p -= lr * p.grad。'],
      ]},
      { name: '8. 常见报错速查', questions: [
        ['“view size is not compatible with input tensor\'s size and stride” 的原因和处理？', '对不连续张量用了 view。改用 .contiguous().view() 或 reshape。'],
        ['“expected scalar type Long but found Float” 的原因？', 'label 必须是 int64，用 targets.long()。'],
        ['“grad can be implicitly created only for scalar outputs” 的原因？', '对非标量调了 backward，loss 要 .mean() 或 .sum()。'],
        ['不报错但 loss 不降，三个常见原因？', '广播错了、忘清梯度、参数没原地更新。先做单 batch 过拟合测试。'],
      ]},
      { name: '9. 出师测验', questions: [
        ['a = [[3,1,2],[0,5,4]]\na - a.mean(dim=1, keepdim=True) = ?', '每行均值 [[2],[3]]，结果 [[1,-1,0],[-3,2,1]]。'],
        ['x = torch.randn(8, 16, 64)\n(x - x.mean(dim=-1, keepdim=True)).shape？这是哪个模块的第一步？', '(8, 16, 64)，LayerNorm 的第一步。'],
        ['动手题：不看手册，20 分钟内从空文件写出完整训练循环，测试准确率 96% 以上。做到了吗？', '做到才算阶段 0 结束，进阶段 1。'],
      ]},
    ],
  },
  {
    name: 'nanoGPT 重现', emoji: '🤖', color: '#6c3fa0',
    chapters: [
      { name: '阶段 1：读懂原版，画形状流转图', questions: [] },
      { name: '阶段 2：盲写 GPT-2', questions: [
        ['阶段 2 的四个验证关卡是什么？', '1. 初始 loss ≈ ln(vocab_size)；2. 单 batch 过拟合到接近 0；3. 因果性测试；4. 加载 HF GPT-2 权重，torch.allclose 对齐。'],
        ['因果性测试怎么做？', '改动位置 t 之后的 token，位置 t 及之前的 logits 必须完全不变。'],
        ['nanoGPT 用的位置编码、归一化、激活函数分别是？', 'learned position embedding、LayerNorm、GELU。没有 RoPE 和 RMSNorm，那些属于 Llama 系。'],
        ['多头注意力里，mask 是怎么加到 attention score 上的？用到了阶段 0 的哪条知识？', '用广播把 (T, T) 的下三角 mask 加到 (B, nh, T, T) 上，未来位置填 -inf 再做 softmax。用到广播的复制规则。'],
      ]},
      { name: '阶段 3：租卡跑真实规模', questions: [] },
      { name: '阶段 4：改造成 Llama 风格', questions: [
        ['阶段 4 要引入哪五个 Llama 组件？', 'RoPE、RMSNorm、SwiGLU、GQA、KV cache。'],
      ]},
    ],
  },
  {
    name: '后训练 / RL 系统', emoji: '🧠', color: '#2a6f97',
    chapters: [
      { name: 'LoRA / QLoRA（LLaMA-Factory / TRL）', questions: [] },
      { name: 'FSDP / DeepSpeed：单机多卡全参训练', questions: [] },
      { name: 'DPO / GRPO（verl / OpenRLHF）', questions: [] },
      { name: 'vLLM / SGLang', questions: [] },
    ],
  },
]};
