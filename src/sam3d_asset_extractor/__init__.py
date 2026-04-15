"""SAM3D asset extractor.

RGB(-D) image -> SAM2 mask -> SAM3D inference -> (optional) mesh decimation.
A preprocessing pipeline that extracts per-object meshes for placement into
a simulator (sim2real data augmentation for VLA fine-tuning).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
