
#TransGAN: Two Pure Transformers Can Make One Strong GAN.

It is a purely Transformer based and no convolution containing a memory friendly generator, a multi-scale discriminator and grid-self attention mechanism. A training technique which includes data augmentation, modifying layer normalization and relative position encoding.


#### Cifar test
First download the [cifar checkpoint](https://drive.google.com/drive/folders/1UEBGHyuDHqr0VzOE9ePx5kZX0zbLqWLh?usp=sharing) and put it on `./cifar_checkpoint`. Then run the following script.
```
python exp/cifar_test.py
```

## Generator and Discriminator model based on Transformer
![Main Pipeline](assets/TransGAN_1.png)

## Representative Visual Results
![Cifar Visual Results](assets/cifar_visual.png)
![Visual Results](assets/teaser_examples.jpg)
