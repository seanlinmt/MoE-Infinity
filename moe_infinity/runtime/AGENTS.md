# moe_infinity/runtime/

## OVERVIEW

Runtime engine for MoE model execution, managing expert offloading and memory-efficient inference.

## WHERE TO LOOK

| Task | File | Key Components |
|------|------|----------------|
| Main offload engine | `model_offload.py` | `OffloadEngine` class (lines 66+), expert loading, cleanup |
| Module init hooks | `hooks.py` | `activate_empty_init()`, `deactivate_empty_init()` for zero-init |
| State dict handling | `state_dict.py` | `partition_offloading_state_dict()`, `load_non_offloading_state_dict()` |
| Integration points | `model_offload.py` | Memory (`moe_infinity.memory`), Models (`moe_infinity.models`), Distributed (`moe_infinity.distributed`) |

## KEY APIS

```python
from moe_infinity.runtime import OffloadEngine

# Initialize offload engine with model config
engine = OffloadEngine(capacity, config)

# Cleanup C++ resources before exit
engine.cleanup()
```