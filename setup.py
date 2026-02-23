#!/usr/bin/env python3
"""Setup script for TextME."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="textme",
    version="1.0.0",
    author="Soyeon Hong, Jinchan Kim, Jaegook You, Seungtaek Choi, Suha Kwak, Hyunsouk Cho",
    author_email="hyunsouk@ajou.ac.kr",
    description="TextME: Bridging Unseen Modalities Through Text Descriptions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/TextME",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "textme-offset=compute_offset:main",
            "textme-train=train:main",
            "textme-eval=evaluate:main",
        ],
    },
)
