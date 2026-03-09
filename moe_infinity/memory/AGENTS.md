# AGENTS.md - Expert Memory Management

## OVERVIEW
Handles expert caching, eviction, and activation-aware prefetching between GPU and host memory.

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Cache logic | `expert_cache.py` | GPU/CPU tiered cache, eviction policy |
| Prefetching | `expert_prefetcher.py` | Activation-aware prefetch using expert matrices |
| Data structures | `expert_entry.py` | `ExpertCacheEntry`, `ExpertTraceEntry` dataclasses |
| Priority scoring | `expert_priority_score.py` | LFU, priority-based eviction scoring |
| Activation tracing | `expert_tracer.py` | Tracks which experts are used per sequence |
| Prediction | `expert_predictor.py` | Predicts future expert activations |

## CONVENTIONS

- **Cache tiers**: Separate `gpu_expert_cache` and `cpu_expert_cache` dicts
- **Protection flags**: Use `experts_protected_ondemand`, `experts_protected_prefetch`, `experts_protected_by_layer` to prevent eviction
- **Policy**: Set via `set_cache_policy("priority")` or `"lfu"`
- **Memory config**: `memory_ratio` multiplied by `gpu_size` for capacity calculation
- **Logging**: Python stdlib `logging` module with stream handler
- **Entry tracking**: `expert_idx`, `layer_idx`, `visit`, `timestamp`, `r` (recency) fields