[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-as-strided-restride-oob-guard

A call-site workaround and diagnostic for a real `torch.compile(backend="inductor")` correctness bug: `torch.as_strided` on a view built from a **repeated + strided-sliced buffer** (re-viewed with the pre-slice stride — a real pattern for weight-tying, broadcasting into a larger strided buffer, or exposing a materialized buffer's full stride after a narrowing slice) makes Inductor compute the **wrong storage-span requirement**, producing either a spurious "out of bounds for storage" `RuntimeError` or a silently wrong result, even though the operation is provably in-bounds by `as_strided`'s own documented `storage_offset + sum((size[i]-1)*stride[i]) + 1` contract. Eager mode always gives the correct, deterministic answer. Upstream references: [pytorch/pytorch#197431](https://github.com/pytorch/pytorch/issues/197431) (open as of this writing) and a related, independently-reported still-open issue describing the same root-cause class, [pytorch/pytorch#192226](https://github.com/pytorch/pytorch/issues/192226).

```python
import torch

def f(x):
    full = x.repeat(3, 2)
    sliced = full[:, ::2]
    return torch.as_strided(sliced, (3, 2), full.stride())

compiled = torch.compile(f, backend="inductor")

x = torch.tensor([[7.0, 9.0]])
print(f(x))          # eager: tensor([[7., 7.], [7., 7.], [7., 7.]])  -- correct, in-bounds
print(compiled(x))   # inductor: RuntimeError: setStorage: ... requiring a storage
                      #           size of 40 are out of bounds for storage of size 24
```

The requested view needs only 11 storage elements (verified by hand against the documented contract) out of `full`'s real storage of 12 — well in-bounds. Inductor's lowering instead computes a completely different, wrong span (40, against a storage it thinks is 24) and raises. Depending on shape, the same defect can instead silently return a wrong, run-to-run-**nondeterministic** result rather than raising at all.

This is not a case of the caller misusing `as_strided` in undefined-behavior territory (unlike the in-place-write-aliasing warning in `as_strided`'s own docs, which only concerns overlapping *writes*, not read correctness) — the operation is provably in-bounds by the tensor's own documented storage/stride math, and eager proves it. The divergence is an internal Inductor lowering defect, not a documented contract violation by the caller.

## Install and check

Requires Python 3.9+ and a compatible PyTorch installation (`torch>=2.0` in the optional extra).

```bash
git clone https://github.com/zhuhroscar-tech/torch-as-strided-restride-oob-guard.git
cd torch-as-strided-restride-oob-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[torch]"
torch-as-strided-restride-oob-guard
torch-as-strided-restride-oob-guard --json
```

The CLI reruns the exact repeat-then-slice-then-restride repro pattern from the upstream issue (across three input-value pairs) against the currently installed torch build, and verifies the `safe_as_strided` guard matches eager on every call. Its JSON includes the installed torch version, per-call results, `any_divergence_bug`, and `guard_fully_correct`.

Exit codes describe the **guard check**, not just native bug detection: `0` means every guard call matched eager, `1` means a guard check failed, and `2` means torch could not be imported.

## Use in Python

```python
from torch_as_strided_restride_oob_guard import safe_as_strided

y = safe_as_strided(sliced, size, stride, storage_offset=None)  # matches torch.as_strided exactly, including under torch.compile
```

`safe_as_strided` is a drop-in replacement for `torch.as_strided` at call sites where the input view has a non-zero `storage_offset` and has already been materialized into its own backing buffer — the pattern that triggers Inductor's storage-span miscomputation. It forces the actual `as_strided` call to run in eager mode via `torch.compiler.disable` — a deliberate, narrow Dynamo graph break at this single, already-cheap op — so the real, documented storage/stride contract is always honored, matching eager exactly whether or not the caller itself is under `torch.compile`.

## Scope and limitations

- This tool does **not** globally monkey-patch `torch.as_strided`. Call `safe_as_strided` explicitly at your own call sites where a repeated/sliced-then-restrided view is built under `torch.compile`.
- The guard mechanism (`torch.compiler.disable`) introduces a graph break at the `as_strided()` call site, which forgoes fusion opportunities for that op. This is a deliberate correctness-over-fusion tradeoff; benchmark before using it on a hot path.
- CPU Inductor is the reproduced and guarded path on this host. CUDA/Triton codegen was not tested here (no CUDA available on this host) — the upstream issue does not specify a device; treat CUDA behavior as unverified by this repo until independently confirmed.
- Only the specific repeat-then-slice-then-restride pattern from the upstream repro is guarded and tested here. `as_strided` calls with other storage-offset/stride combinations may or may not trigger the same Inductor defect; this repo does not attempt to enumerate every triggering shape.
- A plain `as_strided` call with no preceding repeat+slice (e.g. directly on a freshly-allocated tensor) is not demonstrated to be affected by this specific bug; `safe_as_strided` still matches eager there, it is simply unnecessary overhead in that case.
- Diagnostic samples do not prove correctness for every possible input, shape, or invocation order; behavior depends on the installed torch version. If a future torch release fixes pytorch/pytorch#197431 upstream, `any_divergence_bug` should report `False` on that version — `safe_as_strided` remains a safe no-op-equivalent guard in that case.
- Native-language (Chinese/Japanese) community discussion of this specific issue was searched for and not found at acceptance time; the evidence base is the upstream GitHub issue (English) plus this repo's own from-scratch, independently-verified-against-the-documented-contract reproduction.

## Development

```bash
python -m pip install -e ".[dev,torch]"
python -m pytest -v --cov=torch_as_strided_restride_oob_guard
```
