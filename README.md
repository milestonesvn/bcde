# bcde
The Bottleneck Conditional Density Estimator provides a semi-supervised learning framework for high-dimensional conditional density estimation. This repository provides code to run experiments in from the NIPS BDL Workshop contribution [Bottleneck Conditional Density Estimation](http://bayesiandeeplearning.org/papers/BDL_36.pdf). 

This work was done while interning at Adobe Systems.

## Dependencies

Please make sure to install the following dependencies
```
pip install 'Keras==1.1.0'
pip install theano
pip install kaos
pip install numpy
```

## Example
To run the model (2-layer BCDE + factored inference + hybrid training), simply do:
```
python experiment.py
```
It should be easy to play around with the settings in `experiment.py` to achieve the other configurations shown in the paper.
