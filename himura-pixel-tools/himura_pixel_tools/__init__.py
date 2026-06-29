"""Himura Pixel Tools — self-contained local pixel-art game asset generator.

Inspired by PixelLab-like workflows, built from the
``himura_pixel_tools_engineer_spec_no_comfyui.json`` specification.

Layers (ACT architecture):
    A — Desktop web UI  (himura_pixel_tools.desktop)
    C — Control service (himura_pixel_tools.api + himura_pixel_tools.mcp)
    T — Tool runtime     (himura_pixel_tools.runtime + himura_pixel_tools.pixel)

No ComfyUI, no external runtime UI, no cloud generation. Everything runs
locally against Diffusers/PyTorch loaded as a library.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
