# Let's face it

[![Video](https://img.youtube.com/vi/RhazMS4L_bk/maxresdefault.jpg)](https://youtu.be/RhazMS4L_bk)
This repository contains a PyTorch based implementation of the framework for the paper "Let's face it: Probabilistic multi-modal interlocutor-aware generation of facial gestures in dyadic settings.", which was nominated for the Best Paper Award at IVA'20.


## Installations

1. Install Docker

We recommend installing the latest version of the Docker as described [here](https://docs.docker.com/engine/install)

2. Build the Docker image
```
docker build -f containers/glow_Dockerfile . -t lets_face_it_glow
```

3. Setup GPU usage for the docker

By following the instructions in [the official tutorial](https://github.com/NVIDIA/nvidia-docker)



&nbsp;
____________________________________________________________________________________________________________
&nbsp;


## Model training and testing

1. Train the model
```
docker run --gpus 1 -v <path/to/the/dataset>:/data lets_face_it_glow python code/glow_pytorch/train.py code/glow_pytorch/hparams/final_model.yaml
```
where `path/to/the/dataset` should be replaced with the path to the dataset on your machine

Customizing
Most of the model parameters are defined in `code/glow_pytorch/hparams/final_model.yaml`. 
Other configurations are set in `code/config.toml`


## Feature extraction and rendering
Instructions coming in a few days!

## Citation
If you use this code in your research please cite the paper:
```
@inproceedings{jonell2020letsfaceit,
  title={Let's face it: Probabilistic multi-modal interlocutor-aware generation of facial gestures in dyadic settings},
  author={Jonell, Patrik and Kucherenko, Taras and Henter, Gustav Eje  and Jonas Beskow},
  booktitle=={International Conference on Intelligent Virtual Agents (IVA ’20)},
  year={2020},
  publisher = {ACM},
}
```

## Contact
If you encounter any problems/bugs/issues please contact me on Github or by emailing me at pjjonell@kth.se for any bug reports/questions/suggestions. I prefer questions and bug reports on Github as that provides visibility to others who might be encountering same issues or who have the same questions.

