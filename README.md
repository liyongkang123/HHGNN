# HHGNN
This is  for our CIKM2022  paper《Heterogeneous Hypergraph Neural Network for  Friend Recommendation with Human Mobility》

Thank you for your interest in the content of our research. 

Before to execute *HHGNN*, it is necessary to install the following packages:
<br/>
``pip install dgl``
<br/>
``pip install torch``
<br/>
``pip install scikit-learn``
<br/>
``pip install torch-scatter``

## Requirements

- numpy ==1.13.1
- torch ==1.7.1
- scikit-learn==1.0.2
- dgl == 0.7.2
- torch-scatter == 2.0.7

【September 11, 2024】The modified code now supports the latest versions of torch and DGL.

### Basic Usage
 
- --Please run  train.py to train the HHGNN in NYC city.
- Due to the huge number of hyperedges, our algorithm needs to occupy a large storage space of GPU. It probably need at least 18G GPU memory for NYC city.
- You can reduce the GPU memory usage by reducing the number of multi-heads or the dimension of the output layer.
More adjustable hyperparameters in the config.py file

【March 3, 2025】 Please refer to the code implementation in [H3GNN](https://github.com/liyongkang123/H3GNN) to learn more about processing heterogeneous hypergraph data. 
We provide an updated version of the code that can complete the data preprocessing for a single city within 10 minutes.

We welcome your suggestions and encouragement to help us keep improving.
And we urge everyone to contribute to the development of the community together.

We will continue to update and adjust and optimize our algorithm to make it more acceptable to everyone.
