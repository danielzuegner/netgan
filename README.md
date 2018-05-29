# NetGAN: Generating Graphs via Random Walks

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

## References
### Cora dataset
In the `data` folder you can find the Cora-ML dataset originally published by   

McCallum, Andrew Kachites, Nigam, Kamal, Rennie, Jason, and Seymore, Kristie.  
*Automating the construction of internet portals with machine learning.*   
Information Retrieval, 3(2):127–163, 2000.

## Contact
Please contact zuegnerd@in.tum.de in case you have any questions.
