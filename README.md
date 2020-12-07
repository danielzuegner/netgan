# NetGAN: Generating Graphs via Random Walks

<p align="center">
<img src="https://www.in.tum.de/fileadmin//w00bws/daml/netgan/netgan.png" width="400">
</p>

Implementation of the method proposed in the paper:   
**[NetGAN: Generating Graphs via Random Walks](https://arxiv.org/abs/1803.00816)** 

by Aleksandar Bojchevski, Oleksandr Shchur, Daniel Zügner, Stephan Günnemann  
Published at ICML 2018 in Stockholm, Sweden.

Copyright (C) 2018   
Daniel Zügner   
Technical University of Munich   

This implementation is written in Python 3.6 and uses Tensorflow 1.4.1.
## Requirements
Install the reqirements via   
`pip install -r requirements.txt`

Note that the modules `powerlaw` and `python-igraph` are only needed to compute
the graph statistics. If you only want to run NetGAN, feel free to comment out 
the respective parts of the code.

## Run the code
 
 To try our code, the best way to do so is to use the IPython notebook `demo.ipynb`
 
## Pre-trained models used in the paper
Run `graph_generation_pretrained.ipynb` and `link_prediction_pretrained.ipynb` to try our pre-trained models on Cora-ML.
 
## Latent variable interpolation
Run `latent_interpolation.ipynb` to run latent variable interpolation experiments as in the paper.
<p align="center">
<img align="center" src="https://www.in.tum.de/fileadmin/w00bws/daml/netgan/latent_interpolation.png" width="500"/>
</p>

## Installation
To install the package, run `python setup.py install`.

## Citation
Please cite our paper if you use the model or this code in your own work:
```
@inproceedings{DBLP:conf/icml/BojchevskiSZG18,
  author    = {Aleksandar Bojchevski and
               Oleksandr Shchur and
               Daniel Z{\"{u}}gner and
               Stephan G{\"{u}}nnemann},
  title     = {NetGAN: Generating Graphs via Random Walks},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning,
               {ICML} 2018, Stockholmsm{\"{a}}ssan, Stockholm, Sweden, July
               10-15, 2018},
  pages     = {609--618},
  year      = {2018},
}
```

## References
### Cora dataset
In the `data` folder you can find the Cora-ML dataset. The raw data was originally published by   

McCallum, Andrew Kachites, Nigam, Kamal, Rennie, Jason, and Seymore, Kristie. *"Automating the construction of internet portals with machine learning."* Information Retrieval, 3(2):127–163, 2000.

and the graph was extracted by

Bojchevski, Aleksandar, and Stephan Günnemann. *"Deep gaussian embedding of attributed graphs: Unsupervised inductive learning via ranking."* ICLR 2018.

## Contact
Please contact zuegnerd@in.tum.de in case you have any questions.
