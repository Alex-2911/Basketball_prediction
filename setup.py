"""
Setup script for NBA Prediction System CLI

Install with: pip install -e .
This makes the 'nba-predict' command available system-wide.
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nba-prediction-system",
    version="2026.1.0",
    author="NBA Prediction Team",
    description="Advanced NBA betting prediction system with ML and Kelly Criterion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/razi90/Basketball_prediction",
    packages=find_packages(where="2026"),
    package_dir={"": "2026"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.12",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "nba-predict=src.cli:main",
        ],
    },
    include_package_data=True,
)
