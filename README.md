# Information Theoretic Dimensionality Reduction

This is the MMINet dimensionality reduction network implementation, which uses a stochastic mutual information gradient based loss. The aim is to project high-dimensional continuous valued input feature space vectors onto a lower dimensional feature space non-parametrically based on a maximum mutual information (MMI) criterion between the transformed features and their associated discrete class labels. Implementation is in Python using the Chainer deep learning framework. `MMIDimReduction.py` includes the `MMINet` class definition, and `demo.py` demonstrates an example usage. 

# Usage

An example execution is as follows:

```python

from MMIDimReduction import MMINet

model  = MMINet(input_dim = ..., output_dim = ..., net = 'linear')
model2 = MMINet(input_dim = ..., output_dim = ..., net = 'nonlinear')

```

Following model construction, one can learn the linear or non-linear feature transformation based on MMI criterion using training samples, and then reduce dimensionality of input instances.

```python

model.learn(x_train, y_train, num_epochs = 5)
z_train = model.reduce(x_train)
z_test = model.reduce(x_test)

```

Parameter `net = 'linear'` defines the network as a single dense layer architecture with no bias term, whereas `net = 'nonlinear'` defines a multilayer perceptron architecture consisting of two hidden layers with ELU activation functions. The model uses a MomentumSGD optimizer and a maximum number of epochs based stopping criterion by default. These default network implementations are meant to be manipulated later by any arbitrary choice. 

# Paper Citation
If you use this code in your research and find it helpful, please cite the following paper:
> Ozan Ozdenizci, Deniz Erdogmus. "Information Theoretic Feature Transformation Learning for Brain Interfaces". IEEE Transactions on Biomedical Engineering, 2019. https://dx.doi.org/10.1109/TBME.2019.2908099

# Acknowledgments
Our work was partially supported by NSF (IIS-1149570, CNS-1544895, IIS-1715858), DHHS (90RE5017-02-01), and NIH (R01DC009834).
