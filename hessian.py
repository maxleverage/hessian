#!/usr/bin/env python

import math as m
import numpy as np
from keras import backend as K

# Tensorflow
import tensorflow as tf

# Load model weights

def load_model(model, filepath):
    """ Load saved model weights

    """
    return model.load_weights(filepath)


def config_input(model):
    """ Returns a list for use as inputs in instantiating a Keras Jacobian /
    Hessian function
    """
    input_tensors = [model.inputs[0], # input data
                 model.sample_weights[0], # how much to weight each sample by
                 model.targets[0], # labels
                 K.learning_phase()]
    return input_tensors


def create_inputs(data, target):
    """ Returns a list of inputs be fed into a Keras Jacobian /
    Hessian function
    """
    inputs = [data, [1], [target], 0]
    return inputs

def create_grad_fn(input_tensor, gradients):
    """ Instantiates a Keras function mapping in inputs to a list 
    of tensors containing model gradients
    """
    return K.function(inputs=input_tensor, outputs=gradients)


def create_hess_fn(input_tensor, hessian):
    """ Returns a Keras Hessian function mapping evaluating the Hessian matrix
    on the batch of data fed into the function
    """
    return K.function(inputs=input_tensor, outputs=hessian)


def get_gradient(model):
    """ Returns a list of gradient tensors from the model
    
    """
    return K.gradients(model.total_loss, model.trainable_weights)


def get_Keras_Hessian(model, gradients):
    """ Returns a list (len = n.o. params in model) of Jacobians,
    i.e. [[0.8, ...], [0.88, ...], ...]
    - This list can be easily reshaped by numpy to a square matrix representing the Hessian
    evaluated on the batch fed into the Keras Hessian function
    """
    counter = 0
    for grad in gradients:
        grad_unrolled = tf.reshape(grad, shape=[-1, 1])
        if counter == 0:
            jacobian = grad_unrolled
        else:
            jacobian = tf.concat([jacobian, grad_unrolled], 0)
        counter += 1
    dim, _ = jacobian.get_shape()
    print(dim)
    hessian = []
    for i in range(dim):
        print(i)
        dfx_i = tf.slice(jacobian, begin=[i, 0], size=[1, 1])
        ddfx_i = K.gradients(dfx_i, model.trainable_weights)
        counter = 0
        for grad in ddfx_i:
            grad_unrolled = tf.reshape(grad, shape=[-1, 1])
            if counter == 0:
                hess_row = grad_unrolled
            else:
                hess_row = tf.concat([hess_row, grad_unrolled], 0)
            counter += 1
        hessian.append(hess_row)
    return hessian




