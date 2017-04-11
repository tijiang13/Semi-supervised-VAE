import tensorflow as tf
import math

C = - 0.5 * math.log( 2 * math.pi )
def KLD(mu, logvar):
    return - 0.5*(1+logvar-tf.square(mu)-tf.exp(logvar))
def bernoulli(p, x):
    epsilon = 1e-8
    return x * tf.log(p + epsilon) + (1-x) * tf.log(1-p + epsilon)
def gaussian(x, mu, logvar):
    return C - 0.5 * (logvar + tf.square(x - mu) / tf.exp(logvar) )
def std_gaussian(x):
    return C - x**2 / 2
def gaussian_std_margin(mu, logvar):
    return C - 0.5*(tf.square(mu) + tf.exp(logvar))
def gaussian_margin(logvar):
    return C - 0.5*(1 + logvar)
