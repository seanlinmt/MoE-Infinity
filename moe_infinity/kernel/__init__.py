try:
    from .router import (
        launch_fused_softmax_topk,
        launch_fused_softmax_topk_nobias,
    )

    TRITON_AVAILABLE = True
except ImportError:

    def launch_fused_softmax_topk(*args, **kwargs):
        raise NotImplementedError("Triton required. Use GPU/CUDA.")

    def launch_fused_softmax_topk_nobias(*args, **kwargs):
        raise NotImplementedError("Triton required. Use GPU/CUDA.")

    TRITON_AVAILABLE = False
