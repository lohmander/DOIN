#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "torch>=1.12.1",
    "torchvision>=0.13.1",
    "numpy>=1.23.2",
    "Pillow>=9.2.0",
    "clip@git+https://github.com/openai/CLIP.git",
]

test_requirements = []

setup(
    author="Hannes Lohmander",
    author_email="hannes@lohmander.org",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Implementation of the DOIN deep learning framework.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="doin",
    name="doin",
    packages=find_packages(include=["doin", "doin.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/lohmander/doin",
    version="0.1.0",
    zip_safe=False,
)
