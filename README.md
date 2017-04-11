# Semi-supervised learning with Variational Autoencoder

This repo replicates the result in paper [Semi-supervised Learning with Deep Generative Models]() by D.P. Kingma et al.

## Repo structure
```
  M2 model: 
    1 labelled, 499 unlabelled    -> ssl_m2_v1.ipynb
    100 labelled & 100 unlabelled -> ssl_m2_v2.ipynb

  Stacked M1 & M2:
    VAE model by saemundsson      -> ssl_stacked-benchmark.ipynb
    VAE model trained by me       -> ssl_stacked_M1_M2.ipynb
```

## Experiment results(100 labelled, 49900 unlabelled)
+ stacked M1 & M2 ~ 95.7% accuracy in valid & test dataset, using the VAE parameters trained by saemundsson, which could be found [here](https://github.com/saemundsson/semisupervised_vae/tree/master/models)
+ M2, there are two cases, but both cases are not well studied & the performance is far from good.
    1. feed 100 labels & 100 unlabels samples points every iteration.(accuracy~87% in valid & test dataset.)
    2. feed 1 labels, 499 unlabels samples simultaneously.(this strategy is adopted in stacked M1 & M2 model, but seems doesn't function well in pure M2 model.)

For comparison, the classifier used in this model trained with 100 labels data could reach arround 76% accuracy in valid & test dataset.

## analysis & conclusion
For M2 model, it could improve the classifier by a margin of ~ 10%.

Stacked model could improve the classifier by a large margin(~20%), but the result is closely related with the pretrained VAE model.

## Comparison
+ Results in original paper
    1. M2: 11.97(delta = 1.71)
    2. M1+M2: 3.33(delta = 0.14)
+ Mine:
    1. M2: 12 ~ 13
    2. M1+M2: 6.3 ~ 4.3

## Environment

+ Tensorflow 1.01
+ Python 2.7

# Reference
1. [implementation by the author with theano](https://github.com/dpkingma/nips14-ssl)
2. [implementation by saemundsson with tensorflow 0.8](https://github.com/saemundsson/semisupervised_vae)
