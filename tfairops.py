# TensorFlow program to perform basic tensor operations.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
t1 = tf.constant([[5, 10, 15], [4, 8, 12], [6, 12, 18]])
t2 = tf.constant([[5, 5, 5], [2, 2, 2], [3, 3, 3]])
tensor_sum = tf.add(t1, t2)  # Add two tensors.
tensor_sub = tf.subtract(t1, t2)  # Subtract two tensors.
tensor_mul = tf.multiply(t1, t2)  # Multiples two tensors.
tensor_divide = tf.divide(t1, t2)  # Divides two tensors without roundoff.
#tensor_div = tf.div(t1, t2)  # Divides two tensors with roundoff.
#tensor_mod = tf.div(t1, t2)  # Find reminder the modulo operation.
sess = tf.Session()
print(sess.run(t1))
print(sess.run(t2))
print("\n----------- SUM ----------------\n")
print(sess.run(tensor_sum))
print("\n------------ SUBTRACT ----------\n")
print(sess.run(tensor_sub))
print("\n------------ MULTIPLY ----------\n")
print(sess.run(tensor_mul))
print("\n------------ DIVIDE ------------\n")
print(sess.run(tensor_divide))
#print("\n------------ DIVIDE ------------\n")
#print(sess.run(tensor_div))
#print("\n------------ MODULO ------------\n")
#print(sess.run(tensor_mod))
