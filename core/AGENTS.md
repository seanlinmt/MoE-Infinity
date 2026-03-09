# AGENTS.md - core/

## OVERVIEW

C++ async I/O core with threading, memory management, and expert dispatching for host memory offloading.

## STRUCTURE

```
core/
├── aio/         # Async I/O: thread pools, tensor handles, priority handles
├── base/        # Base utilities: threading, logging, timestamps, file ops
├── common/      # Shared types, status, PyTorch interop
├── engine/      # Host-to-device transfer engine
├── memory/      # Device/host caching allocators, memory pools, buffers
├── model/       # Model topology parsing and expert layout
├── parallel/    # Expert dispatcher and module parallelization
├── prefetch/    # Activation-aware prefetching, task scheduling
├── python/      # Python bindings (pybind11)
└── utils/       # Logging, CUDA utils, lockfree structures, tqdm
```

## WHERE TO LOOK

| Task | Location | Key Files |
|------|----------|-----------|
| Async I/O | `aio/` | `archer_aio_threadpool`, `archer_prio_aio_handle` |
| Threading | `base/` | `thread_pool.cc`, `thread.cc`, `countdown_latch` |
| Memory alloc | `memory/` | `device_caching_allocator`, `host_caching_allocator`, `memory_pool` |
| Expert dispatch | `parallel/` | `expert_dispatcher.cpp`, `expert_module.cpp` |
| Prefetching | `prefetch/` | `archer_prefetch_handle`, `task_scheduler` |
| Python binding | `python/` | `py_archer_prefetch.cpp` |

## CONVENTIONS

- Style: Google C++ style (indent-width 2) via `.clang-format` in root
- Build: Custom ops built via `op_builder/`, not CMake directly
- Thread naming: Use `current_thread.h` utilities for identification
- Logging: Dual system - `base/logging.h` and `utils/logger.h`

## ANTI-PATTERNS

- Do not allocate tensors directly; use memory pool allocators
- Do not bypass `archer_prio_aio_handle` for prioritized I/O
- Do not mix `device_caching_allocator` with raw CUDA malloc in hot paths