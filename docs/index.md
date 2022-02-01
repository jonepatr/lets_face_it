---
# redirect_to: "https://patrikjonell.se/projects/lets_face_it/"
---
<h1>Let's Face It</h1>
<h3>Probabilistic Multi-modal Interlocutor-aware Generation of Facial Gestures in Dyadic Settings</h3>

This paper received the Best Paper Award at IVA'20.

<div class="row">
<div class="col-sm-3"><a href="https://doi.org/10.1145/3383652.3423911" class="btn">The Paper</a></div>
<div class="col-sm-3"><a href="https://github.com/jonepatr/lets_face_it" target="_blank" class="btn">Code</a></div>
<div class="col-sm-3"><a href="#data">Data</a></div>
<div class="col-sm-3"><a href="#video-samples">Samples</a></div>
</div>



## Short video summarizing the paper
<div class="video-container">
<iframe height="auto" width=640 src="https://www.youtube.com/embed/RhazMS4L_bk" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<br>
<br>
## Abstract
To enable more natural face-to-face interactions, conversational agents need to adapt their behavior to their interlocutors. One key aspect of this is generation of appropriate non-verbal behavior for the agent, for example facial gestures, here defined as facial expressions and head movements. Most existing gesture-generating systems do not utilize multi-modal cues from the interlocutor when synthesizing non-verbal behavior. Those that do, typically use deterministic methods that risk producing repetitive and non-vivid motions. In this paper, we introduce a probabilistic method to synthesize interlocutor-aware facial gestures -- represented by highly expressive [FLAME](https://flame.is.tue.mpg.de){:target="_blank" rel="noopener"} parameters -- in dyadic conversations. Our contributions are: a) a method for feature extraction from multi-party video and speech recordings, resulting in a representation that allows for independent control and manipulation of expression and speech articulation in a 3D avatar; b) an extension to [MoGlow](https://arxiv.org/abs/1905.06598){:target="_blank" rel="noopener"}, a recent motion-synthesis method based on [normalizing flows](https://arxiv.org/abs/1912.02762){:target="_blank" rel="noopener"}, to also take multi-modal signals from the interlocutor as input and subsequently output interlocutor-aware facial gestures; and c) a subjective evaluation assessing the use and relative importance of the input modalities.
The results show that the model successfully leverages the input from the interlocutor to generate more appropriate behavior.

<h2 id="data" style="padding-top: 60px; ">FLAME facial feature dataset</h2>
Please contact me via email to get access to the dataset.
<br><br>
The dataset is 6.4GB and the features are provided in 25fps. <br>
The data is organized in the following structure: \
`sessions/{1,2...54}/participants/{P1,P2}`
```
tf_exp - expression parameters
tf_pose - neck, eye, and jaw rotation parameters
tf_shape - facial shape parameters
tf_rot - global rotation
tf_trans - global translation
```

<h2 id="video-samples" style="padding-top: 60px;">Video samples</h2>
You can find some video samples that were used in the user studies [here](https://vimeo.com/showcase/7219185).
More video samples with different model settings and sound will be available soon, please see the short summary video in the meantime for some more samples.
<br>
<br>


<h2 id="citing" style="padding-top: 60px;">Citing</h2>
<pre class="citation long">
@inproceedings{jonell2020let,
    author = {Jonell, Patrik and Kucherenko, Taras and Henter, Gustav Eje and Beskow, Jonas},
    title = {Let's Face It: Probabilistic Multi-Modal Interlocutor-Aware Generation of Facial Gestures in Dyadic Settings},
    year = {2020},
    isbn = {9781450375863},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://dl.acm.org/doi/10.1145/3383652.3423911},
    doi = {10.1145/3383652.3423911},
    booktitle = {Proceedings of the 20th ACM International Conference on Intelligent Virtual Agents},
    articleno = {31},
    numpages = {8},
    location = {Virtual Event, Scotland, UK},
    series = {IVA '20}
}
</pre>
