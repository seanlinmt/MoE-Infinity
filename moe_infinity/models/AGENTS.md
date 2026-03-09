# AGENTS.md - moe_infinity/models/

## OVERVIEW

Model implementations with MoE block wrappers that integrate the memory offload system.

## STRUCTURE

```
models/
├── __init__.py              # Model registry (exports MoE blocks)
├── model_utils.py           # Shared rotary embeddings (rotate_half, apply_rotary_pos_emb)
├── deepseek.py              # DeepseekMoEBlock (supports v2 and v3)
├── mixtral.py               # SyncMixtralSparseMoeBlock
├── grok.py                  # SyncGrokMoeBlock
├── arctic.py                # SyncArcticMoeBlock
├── nllb_moe.py              # SyncNllbMoeSparseMLP
├── switch_transformers.py   # SyncSwitchTransformersSparseMLP
├── modeling_deepseek/       # DeepSeek V2 HuggingFace implementation
├── modeling_deepseek_v3/    # DeepSeek V3 HuggingFace implementation
├── modeling_grok/           # Grok-1 implementation
└── modeling_arctic/         # Arctic implementation
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Add new MoE block | Top-level `*.py` files (e.g., `deepseek.py`) |
| Model config classes | Respective `modeling_*/configuration_*.py` |
| Tokenizer classes | Respective `modeling_*/tokenization_*.py` |
| Core model forward | Respective `modeling_*/modeling_*.py` |
| Shared utilities | `model_utils.py` for rotary embeddings |

## ANTI-PATTERNS

- Do not instantiate MoE blocks directly without the offload system - they expect `expert_executor` and `layer_id` to be set by the memory manager
- Do not modify rotary embedding functions without checking all models that use them