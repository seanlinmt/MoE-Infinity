# AGENTS.md

## OVERVIEW
Distributed inference components for multi-GPU/multi-node MoE execution via PyTorch RPC and distributed.

## WHERE TO LOOK
| File | Purpose |
|------|---------|
| `devicemap_manager.py` | Device mapping across nodes using `torch.distributed`, world size planning |
| `expert_executor.py` | Expert dispatch across workers via `torch.distributed.rpc` async calls |
| `expert_prefetcher.py` | Cross-node expert prefetching with RPC coordination |

Key classes: `DeviceMapManager`, `DistributedExpertExecutor`, `DistributedExpertPrefetcher`

## ANTI-PATTERNS
- **DO NOT use for production distributed inference** - This open-source version does NOT support distributed inference (as noted in README)
- These components exist as infrastructure only and are not activated in the current release
- Do not attempt to enable distributed mode via configuration - no such option exists