[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-as-strided-restride-oob-guard

本仓库已迁移到统一的 [`torch-correctness-guards`](https://github.com/zhuhroscar-tech/torch-correctness-guards) 包中。

请改用 umbrella 包：

```bash
git clone https://github.com/zhuhroscar-tech/torch-correctness-guards.git
cd torch-correctness-guards
python -m pip install -e ".[torch]"
torch-guard run as-strided-restride-oob
```

Python API：

```python
from torch_correctness_guards import safe_as_strided
```

迁移后的 guard 保留原用途：诊断 `torch.compile(backend="inductor")` 下 `torch.as_strided` restride 存储跨度计算错误，并提供狭窄的 eager 调用点守卫。
