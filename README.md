Title
================================================================
This repo contains the implementation for the paper Title.

Contents:

- [Installation](#installation)
- [Experiments](#experiments)
    - [Gaussian](#gaussian)
    - [Gaussian Mixture Model](#gmm)
    - [Noisy inpainting](#noisy-inpainting)
- [References](#references)

## Installation
The package requires Python 3.7. First, it is recommended to [create a new python virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands). 
This package depends on JAX. Because the JAX installation is different depending on your CUDA version, this package does not list JAX as a dependency in `setup.py`.
First, [follow these instructions](https://github.com/google/jax#installation) to install JAX with the relevant accelerator support.
Then,
- Clone the repository https://github.com/fedebotu/clone-anonymous-github
- Install using pip `pip install -e .` from the working directory of this README file (see the `setup.py` for the requirements that this command installs).

## Experiments

### Gaussian
Run the example by typing 
```sh
python examples/example.py:
  --config: Training configuration.
    (default: './configs/example.py')
  --workdir: Working directory
    (default: './examples/')
```

### GMM
Run the example by typing 
```sh
python examples/example2.py:
  --config: Training configuration.
    (default: './configs/example2.py')
  --workdir: Working directory
    (default: './examples/')
```

### Noisy Inpainting

First, download the checkpoints and place them into an `exp/` folder in the working directory of this README file.

You will require the pre-trained model checkpoints to access the score models. All checkpoints are from [Score-Based Generative Modeling through Stochastic Differential Equations](https://github.com/yang-song/score_sde/blob/main/README.md) and provided in this [Google drive](https://drive.google.com/drive/folders/1RAG8qpOTURkrqXKwdAR1d6cU9rwoQYnH).

Please note that if for any reason you need to download the CelebAHQ and/or FFHQ datasets, you will need to manually download them using these [instructions](https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training). 

Reproduce our inpainting or superresolution experiments through `main.py`.
```sh
main.py:
  --config: Configuration.
    (default: 'None')
  --eval_folder: The folder name for storing evaluation results
    (default: 'eval')
  --mode: <inpainting|superresolution>: Running mode: inpainting or superresolution
  --workdir: Working directory
```


* `config` is the path to the config file. They are adapted from [https://github.com/yang-song/score_sde/tree/main#:~:text=workdir%3A%20Working%20directory-,config,-is%20the%20path](here). Our prescribed config files are provided in `configs/`.

*  `workdir` is the path that stores all artifacts of one experiment, like checkpoints, which are required for loading the score model.

* `eval_folder` is the name of a subfolder in `workdir` that stores all artifacts of the evaluation process, like image samples, and numpy dumps of quantitative results.

* `mode` is either "inpainting" or "superresolution". When set to "inpainting", it performs an inpainting experiment. When set to "superresolution", it performs the super-resolution experiment.

The experimental setup can be configured through the config file and the experimental parameters within `run_lib.py`.

