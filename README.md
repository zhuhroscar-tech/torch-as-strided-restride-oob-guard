# torch-as-strided-restride-oob-guard

This repository has moved into the consolidated [`torch-correctness-guards`](https://github.com/zhuhroscar-tech/torch-correctness-guards) package.

Use the umbrella package instead:

```bash
git clone https://github.com/zhuhroscar-tech/torch-correctness-guards.git
cd torch-correctness-guards
python -m pip install -e ".[torch]"
torch-guard run as-strided-restride-oob
```

Python API:

```python
from torch_correctness_guards import safe_as_strided
```

The migrated guard keeps the same purpose: diagnose the `torch.compile(backend="inductor")` `torch.as_strided` restride storage-span correctness bug and provide a narrow eager call-site guard.
