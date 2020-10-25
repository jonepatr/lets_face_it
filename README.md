# Let's face it

[![Video](https://img.youtube.com/vi/RhazMS4L_bk/maxresdefault.jpg)](https://youtu.be/RhazMS4L_bk)
This repository contains a PyTorch based implementation of the framework for the paper "Let's face it: Probabilistic multi-modal interlocutor-aware generation of facial gestures in dyadic settings.", which received the Best Paper Award at IVA'20.

Please read more on the [project website](https://jonepatr.github.io/lets_face_it/)


## Installations

1. Install Docker
We recommend installing the latest version of the Docker as described [here](https://docs.docker.com/engine/install)

2. Setup GPU usage for the docker
By following the instructions in [the official tutorial](https://github.com/NVIDIA/nvidia-docker)



&nbsp;
____________________________________________________________________________________________________________
&nbsp;


## Model training and testing

1. Build the Docker image
```
docker build -f containers/glow_Dockerfile . -t lets_face_it_glow
```

2. Train the model
```
docker run --gpus 1 -v <path/to/the/dataset>:/data lets_face_it_glow python code/glow_pytorch/train.py code/glow_pytorch/hparams/final_model.yaml
```
where `path/to/the/dataset` should be replaced with the path to the dataset on your machine

Customizing
Most of the model parameters are defined in `code/glow_pytorch/hparams/final_model.yaml`. 
Other configurations are set in `code/config.toml`


## Visualization
1.  Build the docker
```
docker build -f containers/visualize_Dockerfile . -t lets_face_it_visualize
```

2. Get the models
  * Download FLAME 2019 model from [here](http://flame.is.tue.mpg.de). You need to sign up and agree to the model license for access to the model. Copy the downloaded model inside the `models/flame_model` folder.
  * Download Landmark embedings from [RingNet Project](https://github.com/soubhiksanyal/RingNet/tree/master/flame_model). Copy it inside the `models/flame_model` folder.

3. Run the render server
```
docker run -v $(pwd)/models:/workspace/models -it -p 8000:8000 lets_face_it_visualize
```

4. Try the example code
There is some example code for rendering in `code/examples/visualize_example.py`. This  example assumes that you have downloade the [facial feature dataset](https://kth.box.com/shared/static/tap6b2m3dkxtb447bnmee8nv9uncvzwb.hdf5).
After rendring you will get back a json response from the server with a URL which can be used to access the video.


## Feature extraction
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

