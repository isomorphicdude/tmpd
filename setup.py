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
    description="An API for interfacing grfjax and diffusionjax",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/bb515/ssa",
    author="Benjamin Boys",
    license="MIT",
    packages=find_packages(exclude=['*.test']),
    include_package_data=True,
    # Here i am developing an install strategy,
    # EITHER to make everything up to date install, requiring changes in score_sde code
    # OR make minimum possible changes required to make it work, based around tensorflo==2.7.0 << trying this one first
    # install_requires=[
    #     'numpy',
    #     'scipy',
    #     'h5py',
    #     'matplotlib',
    #     'ml-collections==0.1.0',
    #     'tensorflow-gan==2.0.0',  # not compatible with tensorflow 2.7.0
    #     'tensorflow_io',
    #     'tensorflow_datasets==3.1.0',
    #     'tensorflow==2.4.0',  # not compatible with something, maybe python 3.9, use 2.5.0 - not compatible with numpy 1.20, so use 2.7.0
    #     'tensorflow-addons==0.12.0',  # installed tensorflow-addons-0.20.0
    #     # installed tensorflow-probability==0.20.1  # not compatible with tensorflow 2.7.0, >=2.12, use intead tensorflow-probability==0.15.0
    #     'tensorboard==2.4.0',
    #     'bsl-py==0.10.0',
    #     # TODO: sort out these imports
    #     # 'diffusionjax',
    #     # 'grfjax',
    #     # 'flax==0.3.1',
    #     # 'jax==0.2.18',
    #     # 'jaxlib==0.1.69',
    #     ]
    #
    # This is the tensorflow==2.7.0 build, but I get the following warnings
    # WARNING:tensorflow:From /home/bb515@ad.eng.cam.ac.uk/miniconda3/envs/ssa/lib/python3.9/site-packages/tensorflow_gan/python/estimator/tpu_gan_estimator.py:42: The name tf.estimator.tpu.TPUEstimator is deprecated. Please use tf.compat.v1.estimator.tpu.TPUEstimator instead.
    # WARNING:absl:GlobalAsyncCheckpointManager is not imported correctly. Checkpointing of GlobalDeviceArrays will not be available.To use the feature, install tensorstore.
    # install_requires=[
    #     'ml-collections==0.1.1',
    #     'tensorflow-gan==2.0.0',
    #     'tensorflow_io==0.32.0',
    #     'tensorflow_datasets==4.3.0',
    #     # 'tensorflow==2.7.0',
    #     'tensorflow-addons==0.15.0',  # try to go for a different version with 2.7.0 if else 0.20.0 latest
    #     'tensorflow-probability==0.15.0',
    #     'tensorboard==2.7.0',  # Since it tracks tensorflow 2.7
    #     'numpy',
    #     'scipy',
    #     'h5py',
    #     'matplotlib',
    #     # TODO: sort out these imports
    #     # 'bsl-py==0.10.0',  # this is for score_inverse_problems
    #     # 'diffusionjax',
    #     # 'grfjax',
    #     # 'flax==0.3.3',  # not compatible with diffusionjax
    #     # 'jax==0.2.18',  # not compatible with diffusionjax
    #     # 'jaxlib==0.1.69',  # not compatible with diffusionjax
    #     ]
    # #
    # # Hopefully it is not necessary to do a tensorflow==2.12 version, as this would presumably be more lines of code to change in score_sde
    )
