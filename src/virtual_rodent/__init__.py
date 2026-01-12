"""
VirtualRodent: Neural-to-pose prediction pipeline.

This package provides tools for loading, processing, and modeling
neural signals and 3D pose data from the Virtual Rodent dataset.
"""

__version__ = "0.1.0"
__author__ = "Adam Catto"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "VirtualRodentDataset":
        from virtual_rodent.data.dataset import VirtualRodentDataset
        return VirtualRodentDataset
    elif name == "VirtualRodentDataModule":
        from virtual_rodent.data.datamodule import VirtualRodentDataModule
        return VirtualRodentDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "VirtualRodentDataset",
    "VirtualRodentDataModule",
]
