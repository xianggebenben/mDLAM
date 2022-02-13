# mDLAM: Neural Network Training via monotonous Deep Learning Alternating Minimization 
This is an implementation of monotonous Deep Learning Alternating Minimization(mDLAM) for the neural network training problem, as described in our paper:

Junxiang Wang, Hongyi Li, and Liang Zhao. Accelerated Gradient-free Neural Network Training by Multi-convex Alternating Optimization. (Neurocomputing 2022)


##  Requirements



torch==1.8.1

numpy==1.21.2


## Run the Demo

python mDLAM.py


## Data


Four benchmark datasets Cora, PubMed, Citeseer, and Coauthor-CS are included in this package.


## Cite


Please cite our following paper if you use our MLP code in your own work:

@inproceedings{wang2022mdlam,

author = {Wang, Junxiang, Li, Hongyi and Zhao, Liang},

title = {Accelerated Gradient-free Neural Network Training by Multi-convex Alternating Optimization},

year = {2022},

booktitle = {Neurocomputing},

}
