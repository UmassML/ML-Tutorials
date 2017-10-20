*Welcome to the first Tensorflow tutorial!* This tutorial (amd Tensorflow in general) is best suited for people who already have a decent grasp on how machine learning works.
If you are looking to learn a more beginner friendly approach to machine learning, check out the SciKit Learn tutorials. 

In this demonstration, I will go through a slight variation of Tensorflow's MNIST image recognition code exmaple line by line. The objective is to help you start to get an idea of how Tensorflow is used and to have you understand a basic implementation of a neural net in Tensorflow. 

This is essentially a simpler explanation of https://www.tensorflow.org/get_started/mnist/beginners. If you don't have trouble understanding the original then you may skip to more advanced tutorials. 

Before starting, read the first part of https://www.tensorflow.org/get_started/mnist/beginners where the problem we are trying to solve is described.

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

This tutorial assumes that you have Tensorflow installed already. If not, you can install it with `pip install tensorflow` If you need additional help installing check out https://www.tensorflow.org/install/ 

We import Tensorflow and load the MNIST data set that comes bundled with TF. The parameter `"MNIST_data/"` tells the interpreter to store the data in a local folder called `MNIST_data`.

`one_hot=True` A "one hot" data set can be thought as a set of data where features are represented with arrays of 0's and 1's. I think the idea can best be conveyed with an exmaple. Lets say we had features `['cold', 'warm', 'hot']`. To make this array one-hot we could encode 'cold' as [1,0,0] warm as [0,1,0] and hot as [0,0,1]. So if we had `['cold', 'hot', 'cold', 'warm' 'hot']` under these rules it would be encoded as: 
[[1,0,0],
 [0,0,1],
 [1,0,0],
 [0,1,0],
 [0,0,1]]
 There are many tools that do this encoding for us, so don't dwell too much on this. In our case, the features represented would be the darkness of each pixel.


`input_image = tf.placeholder(tf.float32, [None, 784])`



