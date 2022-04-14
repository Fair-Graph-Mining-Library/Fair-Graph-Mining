# **UVA Fairness Library**
_Contributors: Pranav Bangarbale_
#### _**Dependencies**_
uva-fairness requires:
    Python (>= 3.8)
    NumPy (>= 1.17.3)
    SciPy (>= 1.3.2)
    TensorFlow (>=2.x)
    PyTorch (>= 1.10.x)

#### _**Installation**_
If you already have a working installation of NumPy/SciPy/TensorFlow/PyTorch, the most simple way to install uva-fairness is by:
`pip install uva-fairness`

#### _**Source Code**_
You can download the latest source code with the command:
`git clone https://github.com/Fair-Graph-Mining-Library/uvafairness.git`

#### _**Project History**_
This project was started in 2021 by _Contributor X_, _Contributor Y_, and _Contributor Z_. For more information about these contributors, see [link]. 

#### _**Contact**_
For any questions/comments/bugs/suggestions, please contact rvs4tk@virginia.edu.

#### _**Organization/Structure**_

`src/` is the main directory

`src/data` contains data used to test algorithms

`src/papers/` contains directory corresponding to each fairness paper

`src/papers/paper?/algorithm?/` contains directory corresponding to each fairness algorithm described in the paper

`src/papers/paper?/algorithm?/results/` contains datasets, jupyter notebooks, etc testing each algorithm within the corresponding paper

`src/papers/paper?/algorithm?/implementation` contains python/jupyter files of algorithm implementations

`citations/` contains relevant citations of papers used in this library as well as .pdf files of each involved research paper for further reference

#### _**Papers & Algorithms Included**_

_1. InForm: Individual Fairness in Graph Mining Algorithms_
    https://dl.acm.org/doi/pdf/10.1145/3394486.3403080
    
    PageRank -- Random Walk based algorithm which minimizes two separate terms: 
    smoothing & query specific. Incorporates a regularization parameter, c, in 
    order to balance the two terms.
    See: src/papers/InForm/PageRank/
    
    Spectral Clustering -- Analyzes spectrum of graph laplacian. Finds eigenvectors
    associated with the k smallest eigenvalues.
    See: src/papers/InForm/SpecClust/
    
    LINE -- Learns N X N embedding matrix U in which each node is mapped to d-dimensional 
    vector, embedding structural property
    See: src/papers/InForm/LINE/
    
    

_2. Paper X_
    
    Algo1 -- desc

