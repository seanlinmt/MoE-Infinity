# Stub module for nvtx (NVIDIA Tools Extensions) when pynvtx is not available.
# This provides no-op decorators for profiling without requiring CUDA toolkit.
#
# Usage: Add this directory to PYTHONPATH before importing moe_infinity


class _NVTXStub:
    """Stub class that provides no-op implementations of nvtx functions."""

    @staticmethod
    def annotate(message=None, color=None, category=None):
        """No-op decorator for profiling. Acts as a pass-through."""

        def decorator(func):
            return func

        return decorator


# Create module-level instance for direct usage
annotate = _NVTXStub.annotate


# For compatibility, also expose as a callable class
class annotate_class:
    """Stub annotate class for when nvtx is not available."""

    def __init__(self, message=None, color=None, category=None):
        self.message = message
        self.color = color
        self.category = category

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __call__(self, func):
        return func
