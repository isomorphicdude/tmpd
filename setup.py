"""
Setup script for ssa.

This setup is required or else
    >> ModuleNotFoundError: No module named 'ssa'
will occur.
"""
from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

extra_compile_args = ['-O3']
extra_link_args = []


setup(
    name="ssa",
    version="0.1.0",
    description="An API for interfacing grfjax",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/bb515/ssa",
    author="Benjamin Boys",
    license="MIT",
    packages=find_packages(exclude=['*.test']),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'h5py',
        'matplotlib',
        ]
    )
