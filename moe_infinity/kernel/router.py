import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False

    def _dummy_jit(fn):
        return fn

    triton = type("triton", (), {"jit": _dummy_jit})()


if TRITON_AVAILABLE:

    @triton.jit
    def softmax_kernel(
        output_ptr,
        input_ptr,
        input_row_stride,
        output_row_stride,
        n_rows,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
        num_stages: tl.constexpr,
    ):
        row_start = tl.program_id(0)
        row_step = tl.num_programs(0)
        for row_idx in tl.range(
            row_start, n_rows, row_step, num_stages=num_stages
        ):
            row_start_ptr = input_ptr + row_idx * input_row_stride
            col_offsets = tl.arange(0, BLOCK_SIZE)
            input_ptrs = row_start_ptr + col_offsets
            mask = col_offsets < n_cols
            row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
            row_minus_max = row - tl.max(row, axis=0)
            numerator = tl.exp(row_minus_max)
            denominator = tl.sum(numerator, axis=0)
            softmax_output = numerator / denominator
            output_row_start_ptr = output_ptr + row_idx * output_row_stride
            output_ptrs = output_row_start_ptr + col_offsets
            tl.store(output_ptrs, softmax_output, mask=mask)

    @triton.jit
    def fused_softmax_topk_kernel(
        hidden_ptr,
        weight_ptr,
        bias_ptr,
        routing_mask_ptr,
        routing_weight_ptr,
        B: tl.constexpr,
        H: tl.constexpr,
        E: tl.constexpr,
        TOPK: tl.constexpr,
        BLOCK_E: tl.constexpr,
        normalize_topk: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        off_e = tl.arange(0, BLOCK_E)
        logits = tl.zeros([BLOCK_E], dtype=tl.float32)

        for h in range(0, H):
            hidden_val = tl.load(hidden_ptr + batch_id * H + h)
            weight_row = weight_ptr + h * E + off_e
            w = tl.load(weight_row, mask=off_e < E, other=0.0)
            logits += hidden_val * w

        logits += tl.load(bias_ptr + off_e, mask=off_e < E, other=0.0)
        logits = tl.where(off_e < E, logits, -float("inf"))

        max_logit = tl.max(logits, axis=0)
        numerator = tl.exp(logits - max_logit)

        if normalize_topk:
            denominator = tl.sum(numerator, axis=0)
            softmax = numerator / denominator
        else:
            denominator = tl.sum(numerator, axis=0)
            softmax = numerator / denominator

        topk_weights, topk_indices = tl.topk(softmax, TOPK)

        topk_indices = tl.where(off_e < TOPK, topk_indices, 0)
        topk_weights = tl.where(off_e < TOPK, topk_weights, 0.0)

        tl.store(
            routing_mask_ptr + batch_id * E + off_e,
            tl.cast(off_e < TOPK, tl.int1),
            mask=off_e < E,
        )
        tl.store(
            routing_weight_ptr + batch_id * E + off_e,
            topk_weights,
            mask=off_e < E,
        )

    @triton.jit
    def fused_softmax_topk_nobias_kernel(
        hidden_ptr,
        weight_ptr,
        routing_mask_ptr,
        routing_weight_ptr,
        B: tl.constexpr,
        H: tl.constexpr,
        E: tl.constexpr,
        TOPK: tl.constexpr,
        BLOCK_E: tl.constexpr,
        normalize_topk: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        off_e = tl.arange(0, BLOCK_E)
        logits = tl.zeros([BLOCK_E], dtype=tl.float32)

        for h in range(0, H):
            hidden_val = tl.load(hidden_ptr + batch_id * H + h)
            weight_row = weight_ptr + h * E + off_e
            w = tl.load(weight_row, mask=off_e < E, other=0.0)
            logits += hidden_val * w

        logits = tl.where(off_e < E, logits, -float("inf"))

        max_logit = tl.max(logits, axis=0)
        numerator = tl.exp(logits - max_logit)

        if normalize_topk:
            denominator = tl.sum(numerator, axis=0)
            softmax = numerator / denominator
        else:
            denominator = tl.sum(numerator, axis=0)
            softmax = numerator / denominator

        topk_weights, topk_indices = tl.topk(softmax, TOPK)

        tl.store(
            routing_mask_ptr + batch_id * E + off_e,
            tl.cast(off_e < TOPK, tl.int1),
            mask=off_e < E,
        )
        tl.store(
            routing_weight_ptr + batch_id * E + off_e,
            topk_weights,
            mask=off_e < E,
        )

    def launch_fused_softmax_topk(
        hidden_states,
        weight,
        bias,
        topk,
        normalize_topk=True,
        exec_parallel=None,
    ):
        B, H = hidden_states.shape
        E = weight.shape[1]
        TOPK = topk
        BLOCK_E = 128
        num_stages = 1

        assert E <= 8192
        assert H == weight.shape[0]

        output_mask = torch.zeros(
            (B, E), device=hidden_states.device, dtype=torch.bool
        )
        output_weight = torch.zeros(
            (B, E), device=hidden_states.device, dtype=hidden_states.dtype
        )

        grid = (B,)
        fused_softmax_topk_kernel[grid](
            hidden_states,
            weight,
            bias,
            output_mask,
            output_weight,
            B,
            H,
            E,
            TOPK,
            BLOCK_E,
            normalize_topk,
            num_warps=4,
        )
        return output_mask, output_weight

    def launch_fused_softmax_topk_nobias(
        hidden_states, weight, topk, normalize_topk=True, exec_parallel=None
    ):
        B, H = hidden_states.shape
        E = weight.shape[1]
        TOPK = topk
        BLOCK_E = 128
        num_stages = 1

        assert E <= 8192
        assert H == weight.shape[0]

        output_mask = torch.zeros(
            (B, E), device=hidden_states.device, dtype=torch.bool
        )
        output_weight = torch.zeros(
            (B, E), device=hidden_states.device, dtype=hidden_states.dtype
        )

        grid = (B,)
        fused_softmax_topk_nobias_kernel[grid](
            hidden_states,
            weight,
            output_mask,
            output_weight,
            B,
            H,
            E,
            TOPK,
            BLOCK_E,
            normalize_topk,
            num_warps=4,
        )
        return output_mask, output_weight
else:

    def launch_fused_softmax_topk(*args, **kwargs):
        raise NotImplementedError(
            "Triton required for fused_softmax_topk. Use GPU/CUDA."
        )

    def launch_fused_softmax_topk_nobias(*args, **kwargs):
        raise NotImplementedError(
            "Triton required for fused_softmax_topk_nobias. Use GPU/CUDA."
        )
