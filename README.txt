# Tree Estimator Program
This is a program written in Python that takes an ultrametric tree generated based on DNA from extant taxa, landmark coordinates of all leaf nodes, computes a new phylogenetic tree with branch lengths representing morphological change between taxa.


### Libraries
Python libraries required to run the program:
ete3
numpy
scipy


### Packages installation
Install Python
https://www.python.org/downloads/

Install Numpy and Scipy
https://www.scipy.org/scipylib/download.html

Install ete3
http://etetoolkit.org/download/


## How to run the program
Command line options:
-n: takes the original newick file and landmark coordinates (all coordinates are in a folder)
python fossil.py original_tree.nw landmark −n

-v: takes the new tree that represents morphological change and verifies its correctness
python fossil .py original tree .nw landmark −v

-f: fits a fossil to an existing tree, takes a tree in newick format (the original ultrametric tree), the landmark file, and the landmark coordinates of the fossil to be fit in the tree
python fossil.py original_tree.nw landmark foreign.txt −f


## Author
Student: Li Li
Advisor: Nina Amenta, Tim Weaver
