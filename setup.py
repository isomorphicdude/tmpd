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
    name="tmpd",
    version="0.0.0",
    description="tmpd is a diffusion package for solving linear inverse problems",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(exclude=['*.test']),
    include_package_data=True,
    install_requires=[
        'ml-collections==0.1.1',
        'tensorflow-gan==2.0.0',
        'tensorflow-io=0.32.0',
        'tensorflow_datasets==4.3.0',
        'tensorflow-probability==0.15.0',
        'tensorboard==2.7.0',
        'numpy',
        'scipy',
        'h5py',
        'matplotlib',
        'absl-py==0.10.0',
        'flax==0.3.3',
        'diffusionjax',
         ]
    )
