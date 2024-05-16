# Copyright 2023 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os

from setuptools import find_packages, setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


long_description = read("README.md")

setup(
    name="unisim",
    version=get_version("unisim/__init__.py"),
    description="UniSim: Universal Similarity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Google",
    author_email="unisim@google.com",
    url="https://github.com/google/unisim",
    license="MIT",
    install_requires=["tabulate", "numpy", "tqdm", "onnx", "jaxtyping", "onnxruntime-gpu", "pandas", "usearch>=2.6.0"],
    extras_require={
        "tensorflow": ["tensorflow>=2.11"],
        "dev": [
            "datasets",
            "mypy",
            "pytest",
            "flake8",
            "pytest-cov",
            "twine",
            "tabulate",
            "black",
            "isort",
            "tf2onnx",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
    packages=find_packages(),
)
