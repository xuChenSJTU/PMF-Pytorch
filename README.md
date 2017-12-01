# PMF-Pytorch
******************************This the implementation of PMF using Pytorch***********************************
*This is a Pytorch code for probabilistic matrix factorization using Adam update rules in recommendation.*\
*All files are organized and they are easy to be understood*\
*You can use movielen-1m for testing this code. Please note the data path in this code are all relative path.*\
The files are following:\
===>1. 0.data_process-1.py\
Generate data for pmf_main.py file\
===>2. PMF_main.py\
The main file of pmf algorithm, define some hyper-parameters.\
===>3 PMF_model.py\
This file contains the main pmf model definition.\
===>4 evaluations.py\
This file defines the evaluation metric way for this algorithm.(RMSE in this file)\
\
Runing Note:\
0.data_process-1.py ---> PMF_main.py\
\
paper reference:\
Probabilistic Matrix Factorization\
http://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf
