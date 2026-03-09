# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-10
**Commit:** 7123e01
**Branch:** fix-host-memory-ratio

## OVERVIEW
MoE-Infinity: cost-effective, fast Mixture-of-Experts (MoE) inference library with PyTorch + HuggingFace. Offloads experts to host memory with activation-aware prefetching/caching.

## STRUCTURE
```
MoE-Infinity/
├── core/                    # C++ async I/O, threading, memory management
├── moe_infinity/            # Main Python package
│   ├── models/              # Model implementations (DeepSeek, Mixtral, etc.)
│   ├── memory/              # Expert memory management
│   ├── distributed/         # Distributed inference
│   ├── runtime/             # Runtime components
│   ├── ops/                 # Operations (links to core/)
│   ├── entrypoints/         # API entrypoints (OpenAI server)
│   ├── common/              # Common utilities
│   └── utils/               # Utilities
├── op_builder/              # Custom operator builders
├── examples/                # Usage examples
└── tests/                   # Test files
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add new model | `moe_infinity/models/` | Follow existing model_*.py patterns |
| Memory/offload | `moe_infinity/memory/` | Expert caching logic |
| API server | `moe_infinity/entrypoints/openai/` | FastAPI endpoints |
| C++ core | `core/` | Async I/O, thread pools |
| Custom ops | `op_builder/` | Operator building |

## CONVENTIONS
- **Python**: ruff (line-length 80), isort, pre-commit hooks
- **C++**: clang-format (Google style, indent-width 2)
- **Build**: `BUILD_OPS=1 pip install -e .` for C++ extensions
- **Version**: `MOEINF_VERSION` env var
- **offload_path**: Must be unique per model

## ANTI-PATTERNS (THIS PROJECT)
- DO NOT reuse `offload_path` across different MoE models
- DO NOT use distributed inference (not supported in this version)
- `offload_path` must be on SSD for performance

## UNIQUE STYLES
- Symlink structure: `moe_infinity/ops/core/core` → `core/`
- Custom operator building via `op_builder/`
- Activation-aware expert prefetching in `moe_infinity/memory/`

## COMMANDS
```bash
# Install
pip install -e .

# Install with C++ ops
BUILD_OPS=1 pip install -e .

# Pre-commit
pre-commit run --all-files

# Run OpenAI server
python -m moe_infinity.entrypoints.openai.api_server --model <model> --offload-dir <path>

# Run example
python examples/interface_example.py --model_name_or_path <model> --offload_dir <path>
```

## NOTES
- Open-source version differs from paper (SLA not priority)
- Supports: DeepSeek-V2, Switch-Transformers, NLLB-MoE, Mixtral
- FlashAttention auto-integrated if installed
- 29 TODO items, 19 FIXME items in codebase