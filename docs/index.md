## Let's Face It: Probabilistic Multi-modal Interlocutor-aware Generation of Facial Gestures in Dyadic Settings
[Patrik Jonell](http://www.patrikjonell.se/), [Taras Kucherenko](https://svito-zar.github.io/), [Gustav Eje Henter](https://people.kth.se/~ghe/), and [Jonas Beskow](https://www.kth.se/profile/beskow)

## Short video summarizing the paper

[![Video](https://img.youtube.com/vi/RhazMS4L_bk/maxresdefault.jpg)](https://youtu.be/RhazMS4L_bk)

## Abstract
To enable more natural face-to-face interactions, conversational agents need to adapt their behavior to their interlocutors. One key aspect of this is generation of appropriate non-verbal behavior for the agent, for example facial gestures, here defined as facial expressions and head movements. Most existing gesture-generating systems do not utilize multi-modal cues from the interlocutor when synthesizing non-verbal behavior. Those that do, typically use deterministic methods that risk producing repetitive and non-vivid motions. In this paper, we introduce a probabilistic method to synthesize interlocutor-aware facial gestures — represented by highly expressive [FLAME](https://flame.is.tue.mpg.de) parameters — in dyadic conversations. Our contributions are: a) a method for feature extraction from multi-party video and speech recordings, resulting in a representation that allows for independent control and manipulation of expression and speech articulation in a 3D avatar; b) an extension to [MoGlow](https://arxiv.org/abs/1905.06598), a recent motion-synthesis method based on [normalizing flows](https://arxiv.org/abs/1912.02762), to also take multi-modal signals from the interlocutor as input and subsequently output interlocutor-aware facial gestures; and c) a subjective evaluation assessing the use and relative importance of the input modalities.
The results show that the model successfully leverages the input from the interlocutor to generate more appropriate behavior.

## The Paper
[The Paper](https://github.com/jonepatr/lets_face_it/raw/master/paper/jonell_lets_face_it.pdf)

## Code
[Github repo](https://github.com/jonepatr/lets_face_it)

## FLAME Facial feature dataset
Coming soon!

## Video samples
You can find some video samples that were used in the user studies [here](https://vimeo.com/showcase/7219185).
More video samples with different model settings and sound will be available soon, please see the short summary video in the meantime for some more samples. 

## Citing
```
@inproceedings{10.1145/3383652.3423911,
author = {Jonell, Patrik and Kucherenko, Taras and Henter, Gustav Eje and Beskow, Jonas},
title = {Let's Face It: Probabilistic Multi-Modal Interlocutor-Aware Generation of Facial Gestures in Dyadic Settings},
year = {2020},
isbn = {9781450375863},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi-org.focus.lib.kth.se/10.1145/3383652.3423911},
doi = {10.1145/3383652.3423911},
booktitle = {Proceedings of the 20th ACM International Conference on Intelligent Virtual Agents},
articleno = {31},
numpages = {8},
location = {Virtual Event, Scotland, UK},
series = {IVA '20}
}
```
