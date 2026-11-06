# Integer-coordinate symbol / soft-H census

[`integer-coordinate-census.py`](integer-coordinate-census.py) 读取原始
`6m-q4-fp32.tch` 的 **429 个 FP32 参数**与 immutable `epoch-02-z.pt`。
输入仅为 tensor dictionaries，矩阵按序独立处理；训练器、模型构造、kernel、
teacher、训练数据均保持原状。目标是解释首次 hard change 的确定性符号账与熵几何。

## 冻结口径

```text
s  = original_scale.to(BF16).to(FP32)       # 保留符号与 BF16 bits
q0 = round(original_weight / s).clamp(-7,7)
q2 = round(saved_z).clamp(-7,7)
Hsoft(z) = -N sum_c p_c log(max(p_c,1e-12))
p_c = mean_i softmax_c(-(z_i-c)^2 / 0.5), c=-7,...,7
g0 = grad_z [2 Hsoft(z) / 586459020] at z=q0
g2 = grad_z [2 Hsoft(z) / 586459020] at z=saved_z
dz = saved_z-q0; dq = q2-q0; dW = dq*s
```

输出校验 **111 矩阵 / 5,868,864 符号 / 5,923,228 原参数**。hard entropy
为逐矩阵经验直方图的 `-sum n_c log(n_c/N)`，全局值为矩阵值之和。
`encoded_W` 指实际导出的 hard 符号乘原 signed BF16 scale。
static15 的 4096-total 频率归一化、header、range rounding 与完整 W 字节
由原 recoder 实测；本报告的 hard entropy 对应理想码长诊断。

**H gradient 全部使用 frozen FP32 autograd 与 `2/FULL_TARGETS` 权重**。
报告保存 norm、max |g|、`|g|<1e-8` 元素比例，以及与 `dz` / `dq` 的 dot
和归一化 cosine。正 dot 表示该端点 H 梯度方向上的位移分量；负 dot 表示
局部下降分量；零 norm 的 cosine 为 JSON `null`。Adam 逐步归因需加入同一
轨迹的 CE/KL 梯度、动量、二阶矩与 projection。此脚本提供两个端点的熵几何。

## 精确 row 加和

对每个矩阵、每个符号，设质量从 `m0` 线性变到 `m2`，定义
`L(m)=-m log(max(m/N,clamp))`。质量有变化时取 secant
`c=[L(m2)-L(m0)]/(m2-m0)`；质量相同时取该点导数。row 的贡献为
`sum_c (m2_row-m0_row)*c`，所有 row 之和等于该矩阵的 ΔH。
hard clamp 为 `1e-30`，soft clamp 为冻结的 `1e-12`。

soft 归因复用 **FP32 softmax assignments、FP64 质量加和**，以保持 row 账
精确加和；`trainer_soft_entropy_nats_fp32` 同时保留训练器原 FP32 scalar。
两种 reduction 的差额体现浮点加和精度。两组 H gradient 均沿用训练器 FP32 scalar。

- matrix/global 的原→新 transition 保存完整 **15×15** 计数。
- `rows` 包含全部 hard-changed row、全部 `|s|<1e-10` tiny row，以及全局
  top contributor。row transition 为非零 `[from,to,count]` 三元组，含对角。
- fractional-only row 参与所有 matrix/global soft ΔH、gradient 和 top 排序；
  `omitted_rows_soft_entropy_delta_nats_sum` 保存省略明细的余项。
- row 的 `H_original` / `H_coordinates` 为六列向量，列名在
  `definitions.H_gradient_row_columns`；matrix/global 使用具名字段。
- `soft_hard_sign_cross` 的行轴为 hard ΔH，列轴为 soft ΔH，顺序为
  decrease / zero / increase；包含矩阵数、row 数、元素数和 changed-symbol 权重。
- `probabilities` 为逐矩阵 15-bin hard/soft 概率，全局按矩阵元素数加权。
  全局 entropy 继续使用逐矩阵之和。soft 概率加和的浮点余差单独记录。
- `top_matrices` / `top_rows` 按贡献绝对值排序并保留正负值，默认各 20 条；
  entropy 与四组 gradient dot 分别排序。

## 运行与最小检查

从 Hutter 仓库目录，用已安装 PyTorch 的 CPU Python 执行一次：

```sh
python3 -B research/integer-coordinate-census.py \
  --checkpoint models/6m-q4-fp32.tch \
  --coordinates /YOUR/temp/trajectory/epoch-02-z.pt \
  --output /YOUR/temp/round413/integer-coordinate-census-e02.json \
  --self-check
```

`--self-check` 先运行一个人工 **3×4** 矩阵：四个已知 hard transitions、八个
对角符号、signed tiny scales、hard/soft row 熵加和、加权梯度 finite / 权重一致，
以及整合报告的 JSON round trip。检查与正式 census 共享归因及汇总代码。
单独检查：

```sh
python3 -B research/integer-coordinate-census.py --self-check \
  --output /YOUR/temp/round413/integer-coordinate-census-synthetic.json
```

输入 ZIP checkpoint 用 `torch.load(..., weights_only=True, mmap=True)` 映射，
torch intra/inter-op 各一线程。每矩阵结束检查 peak RSS **≤900,000,000 B**；
超额时停止。JSON 使用紧凑格式，记录输入/source path/bytes、CPU/wall、torch 版本
和 peak RSS。训练轮数、tau、熵权重、optimizer 与导出方法保持冻结。
