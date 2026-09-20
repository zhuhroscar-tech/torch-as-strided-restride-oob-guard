[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-as-strided-restride-oob-guard

针对 `torch.compile(backend="inductor")` 一个真实正确性缺陷的调用点规避方案与诊断工具：对由「重复（repeat）+ 跨步切片（strided slice）」构造的视图（再用切片前的 stride 重新 `as_strided` —— 这是权重绑定、把小张量广播进更大跨步缓冲区、或在收窄切片后重新暴露已物化缓冲区完整 stride 的真实场景）调用 `torch.as_strided` 时，Inductor 会计算出**错误的存储跨度需求**，从而抛出伪造的「越界」`RuntimeError`，或返回一个静默错误的结果 —— 尽管该操作按照 `as_strided` 自身文档化的 `storage_offset + sum((size[i]-1)*stride[i]) + 1` 契约是可证明地在界内的。Eager 模式始终给出正确、确定的结果。上游参考：[pytorch/pytorch#197431](https://github.com/pytorch/pytorch/issues/197431)（截至撰写时仍未关闭）以及一个描述同一根因类别、独立报告且仍未修复的相关 issue [pytorch/pytorch#192226](https://github.com/pytorch/pytorch/issues/192226)。

```python
import torch

def f(x):
    full = x.repeat(3, 2)
    sliced = full[:, ::2]
    return torch.as_strided(sliced, (3, 2), full.stride())

compiled = torch.compile(f, backend="inductor")

x = torch.tensor([[7.0, 9.0]])
print(f(x))          # eager: tensor([[7., 7.], [7., 7.], [7., 7.]])  -- 正确，在界内
print(compiled(x))   # inductor: RuntimeError: setStorage: ... requiring a storage
                      #           size of 40 are out of bounds for storage of size 24
```

该视图请求实际只需要 11 个存储元素（已按文档化契约手工验证），而 `full` 的真实存储有 12 个 —— 完全在界内。Inductor 的 lowering 却计算出一个完全不同的、错误的跨度（40，针对它认为是 24 的存储）并抛出异常。根据具体形状不同，同一缺陷也可能不抛出异常，而是静默返回一个逐次运行结果不一致的错误值。

这并非调用方在未定义行为区域误用 `as_strided`（不同于 `as_strided` 文档中关于原地写入别名的警告，那只涉及*写入*重叠，与读取正确性无关）—— 该操作按张量自身文档化的存储/stride 数学是可证明在界内的，eager 已经证明了这一点。这一偏差是 Inductor 内部 lowering 的缺陷，而非调用方违反文档化契约。

## 安装与检查

需要 Python 3.9+ 及兼容的 PyTorch 安装（可选依赖中的 `torch>=2.0`）。

```bash
git clone https://github.com/zhuhroscar-tech/torch-as-strided-restride-oob-guard.git
cd torch-as-strided-restride-oob-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[torch]"
torch-as-strided-restride-oob-guard
torch-as-strided-restride-oob-guard --json
```

该 CLI 会针对当前安装的 torch 版本，重新执行上游 issue 中「重复 -> 切片 -> 重新 restride」的复现模式（覆盖三组输入值），并验证 `safe_as_strided` 守卫在每次调用中都与 eager 结果一致。其 JSON 输出包含已安装的 torch 版本、逐次调用结果、`any_divergence_bug` 与 `guard_fully_correct`。

退出码描述的是**守卫检查**结果，而非仅仅是原生缺陷检测：`0` 表示每次守卫调用都与 eager 一致，`1` 表示某次守卫检查失败，`2` 表示无法导入 torch。

## 在 Python 中使用

```python
from torch_as_strided_restride_oob_guard import safe_as_strided

y = safe_as_strided(sliced, size, stride, storage_offset=None)  # 与 torch.as_strided 完全一致，包括在 torch.compile 下
```

`safe_as_strided` 是 `torch.as_strided` 的直接替代品，适用于输入视图具有非零 `storage_offset` 且已被物化为自身独立缓冲区的调用点 —— 正是触发 Inductor 存储跨度误算的模式。它通过 `torch.compiler.disable` 强制实际的 `as_strided` 调用在 eager 模式下运行 —— 这是在该单一、已经很廉价的算子调用点上有意为之的、狭窄的 Dynamo 图断裂 —— 因此始终遵循真实、文档化的存储/stride 契约，无论调用方本身是否处于 `torch.compile` 之下都与 eager 完全一致。

## 范围与局限性

- 本工具**不会**全局猴子补丁 `torch.as_strided`。请在你自己构造「重复/切片后重新 restride」视图且处于 `torch.compile` 下的调用点显式调用 `safe_as_strided`。
- 守卫机制（`torch.compiler.disable`）会在 `as_strided()` 调用点引入图断裂，从而放弃该算子的融合机会。这是刻意为之的「正确性优先于融合」权衡；在热路径上使用前请先做基准测试。
- 本机复现与守卫验证的是 CPU Inductor 路径。本机没有 CUDA，未测试 CUDA/Triton 代码生成 —— 上游 issue 未指明具体设备；在本仓库独立验证之前，请将 CUDA 行为视为未经验证。
- 本仓库仅守卫并测试了上游复现中「重复 -> 切片 -> 重新 restride」这一具体模式。其他 storage-offset/stride 组合的 `as_strided` 调用是否会触发同一 Inductor 缺陷尚不确定；本仓库不试图枚举所有可能触发的形状。
- 未经过「先重复+切片」步骤、直接在新分配张量上调用的普通 `as_strided`，并未被证明受此特定缺陷影响；此时 `safe_as_strided` 仍与 eager 一致，只是多余的开销。
- 诊断样例并不能证明所有可能的输入、形状或调用顺序下的正确性；具体行为取决于已安装的 torch 版本。若未来某个 torch 版本在上游修复了 pytorch/pytorch#197431，则该版本上 `any_divergence_bug` 应报告为 `False` —— 此时 `safe_as_strided` 仍是一个安全的、等效于空操作的守卫。
- 在采纳该候选项时，已搜索但未找到该具体问题的中文/日文社区讨论；证据基础是上游 GitHub issue（英文）加上本仓库自身从零开始、并已独立对照文档化契约验证过的复现。

## 开发

```bash
python -m pip install -e ".[dev,torch]"
python -m pytest -v --cov=torch_as_strided_restride_oob_guard
```
