#!/usr/bin/env python
"""
P3-SAM: Native 3D Part Segmentation
Setup script for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="p3sam",
    version="1.0.0",
    author="Changfeng Ma, Yang Li, Xinhao Yan, et al.",
    author_email="",
    description="P3-SAM: Native 3D Part Segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tencent/Hunyuan3D-Part",
    # Map P3-SAM directory to p3sam package
    package_dir={"p3sam": "."},
    packages=["p3sam", "p3sam.demo", "p3sam.utils"],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "trimesh>=3.9.0",
        "scikit-learn>=1.0.0",
        "fpsample>=0.1.0",
        "tqdm>=4.60.0",
        "numba>=0.55.0",
        "safetensors>=0.3.0",
        "Pillow>=9.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "demo": [
            "viser>=0.1.0",
            "gradio>=3.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "p3sam-auto-mask=demo.auto_mask:main",
            "p3sam-app=demo.app:main",
        ],
    },
    package_data={
        "": ["*.md", "*.txt", "*.yaml", "*.json"],
    },
    include_package_data=True,
    zip_safe=False,
)

