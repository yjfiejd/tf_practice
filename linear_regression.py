# @TIME : 2018/12/21 上午8:32
# @File : linear_regression.py


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd
import os
from sklearn.utils import check_random_state

# Generating artificial data.
n = 50
XX = np.arange(n)
rs = check_random_state(0)
YY = rs.randint(-10, 10, size=(n,)) + 2.0 * XX
data = np.stack([XX,YY], axis=1)


# defining flags
num_epochs = 50

# creating the weight and bias
W = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')


# function
# creating placeholders for input X and label Y --> placeholder
#  Creating placeholders for input X and label Y.
def inputs():
    """
    Defining the place_holders.
    :return:
            Returning the data and label place holders.
    """
    X = tf.placeholder(tf.float32, name="X")
    Y = tf.placeholder(tf.float32, name="Y")
    return X,Y

# Create the prediction.
def inference(X):
    """
    Forward passing the X.
    :param X: Input.
    :return: X*W + b.
    """
    return X * W + b

def loss(X, Y):
    '''
    compute the loss by comparing the predicted value to the actual label.
    :param X: The input.
    :param Y: The label.
    :return: The loss over the samples.
    '''

    # Making the prediction.
    Y_predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))/(2*data.shape[0])


# The training function.
def train(loss):
    learning_rate = 0.0001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:

    # Initialize the variables[w and b].
    sess.run(tf.global_variables_initializer())

    # Get the input tensors
    X, Y = inputs()

    # Return the train loss and create the train_op.
    train_loss = loss(X, Y)
    train_op = train(train_loss)

    # Step 8: train the model
    for epoch_num in range(num_epochs):
        loss_value, _ = sess.run([train_loss,train_op],
                                 feed_dict={X: data[:,0], Y: data[:,1]})

        # Displaying the loss per epoch.
        print('epoch %d, loss=%f' %(epoch_num+1, loss_value))

        # save the values of weight and bias
        wcoeff, bias = sess.run([W, b])



Input_values = data[:,0]
Labels = data[:,1]
Prediction_values = data[:,0] * wcoeff + bias

plt.plot(Input_values, Labels, 'ro', label='main')
plt.plot(Input_values, Prediction_values, label='Prediction')

plt.legend()
plt.show()
plt.close()
