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


```python 
input_image = tf.placeholder(tf.float32, [None, 784])
```

In TF, a placeholder is a variable that will have a value later when we run a `Session`. In the parameters of `tf.placeholder` we describe the data type and the dimensionality of the placeholder. We want it to be of type float so we can execute matrix multiplication on it later. Since we want this placeholder to contain a vector representation of our input images, we assign the first dimension to `None` which means that the dimension can be any length. In other words, we can pump as many images as we want into the learning algorithm, and we assign the second vector to 784 dimensions, one for each pixel of the 28x28 image. 

```python
weights = tf.Variable(tf.zeros([784, 10]))
```

Here we declare a `tf.Variable`. A Variable is a value that can be changed by Tensorflow (and subsequently our learning algorithm). The parameter for a Variable is its initial value. tf.zeroes will fill it up with 0's in the dimensions specified. 784 for the 784 initial neurons, and 10 output neurons.

```python
bias = tf.Variable(tf.zeros([10]))
```

Instantiate ten bias nodes, one for each layer of weights.

```python
softmax_result = tf.nn.softmax(tf.matmul(input_image, weights) + bias)
```

Now it's time to implement our learning model. By using TF's built-in matrix multiplication function, we create the synapse between our input and output nodes. Tensorflow has built-in normalization functions. We use softmax regression here since we have multiple defined classes of output. (0-9) 

```python
prediction = tf.placeholder(tf.float32, [None, 10])
```

Here we declare a placeholder for our output neurons, of size 10, one for each of our output classes (the numbers 0-9).

```python
cross_entropy = tf.reduce_mean(-tf.reduce_sum(prediction * tf.log(softmax_result), reduction_indices=[1]))
```

Now we're ready to define our loss function. This tutorial chose to implement the cross entropy loss function. 



