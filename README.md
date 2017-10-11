This is an implementation of [The Cramer Distance as a Solution to Biased Wasserstein Gradients](https://arxiv.org/abs/1705.10743) in [Chainer](https://github.com/chainer/chainer) v3.0.0rc1.

# Requirements
Chainer v3.0.0rc1, OpenCV, etc.  
The scripts work on Python 2.7.13.

# How to generate images
```
$ python generate_image.py example_food-101/config.py -p example_food-101/trained-params_gen_update-000040000.npz
```
You can generate various images by changing the random_seed option.
```
$ python generate_image.py example_food-101/config.py -r 1 -p example_food-101/trained-params_gen_update-000040000.npz
```

## Example Food-101

# Dataset
I resized the images to 64x64 before training.
* [Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)  
Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc. Food-101 -- Mining Discriminative Components with Random Forests. European Conference on Computer Vision, 2014.
