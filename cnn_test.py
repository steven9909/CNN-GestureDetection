from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import random
import tensorflow as tf



def placeholders(n_H, n_W, n_C, numClasses):
    batch = tf.placeholder(tf.float32, [None, n_H, n_W, n_C])
    output = tf.placeholder(tf.float32, [None, numClasses])

    return batch, output


def model(X_train, X_ans, test_train, test_ans,learning_rate=0.0000001, num_epoch=5000):

    (m, n_H, n_W, n_C) = X_train.shape

    numClasses = X_ans.shape[1]

    batch, output = placeholders(n_H, n_W, n_C, numClasses)

    filters = initFilters()

    propogated = propogateForward(batch, filters)

    cost = get_cost(propogated, output)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    initialize = tf.global_variables_initializer()

    with tf.Session() as sess:
       
        sess.run(initialize)

        for epoch in range(num_epoch):
            index = random.randint(0, m-6)
            _, c = sess.run([optimizer, cost], feed_dict={
                            batch: X_train[index:index+5], output: X_ans[index:index+5]})

            #if epoch % 5 == 0:
                #print("Cost is: %f" % (c))

        predict = tf.argmax(propogated, 1)
        correct = tf.equal(predict, tf.argmax(output, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, "float"))

        train_accuracy = accuracy.eval({batch: X_train, output: X_ans})

        test_accuracy = accuracy.eval({batch:test_train, output: test_ans})

        save_path = saver.save(sess, "/tmp/model.ckpt")
        
        print("Model saved in path: %s" % save_path)
        
        print("Train accuracy: ", train_accuracy)

        print("Test accuracy: ", test_accuracy)


def get_cost(layer, label):

    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer, labels=label))
    cost = tf.reduce_mean(-tf.reduce_sum(label * tf.log(layer)))
    return cost


def propogateForward(X, filters):

    conv1 = tf.nn.conv2d(X, filters, strides=[1, 1, 15, 1], padding="SAME")

    conv1Activate = tf.nn.relu(conv1)

    pool = tf.nn.max_pool(conv1Activate, ksize=[1, 1, 4, 1], strides=[
                          1, 1, 4, 1], padding="SAME")

    poolActivate = tf.nn.relu(pool)

    flat = tf.contrib.layers.flatten(poolActivate)

    dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense, rate=0.4)

    logits = tf.layers.dense(inputs=dropout, units=2, activation=tf.nn.softmax)

    return logits


def initFilters():

    filters = tf.get_variable(
        "F1", [1, 15, 3, 128], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    return filters

def convertTextToData(accel_data_file,accel_ans_file):

    array = accel_ans_file.read().split(',')

    array2 = accel_data_file.read().split('~')

    data = []

    for counter in range(len(array2)):
        y = np.zeros((1, 200, 3))

        temp = array2[counter]
        temp2 = temp.split(",")

        for i in range(200):
            coordinate = temp2[i]
            indv = coordinate.split(";")
            for j in range(3):
                y[0][i][j] = indv[j]

        data.append(y)

    data = np.asarray(data,dtype=np.float32)

    numSamples = data.shape[0]

    for i in range(len(array)):
        if (array[i] == "OnTheGround"):
            array[i] = "0"
        elif (array[i] == "UpAndDown"):
            array[i] = "1"

    answers = np.zeros((numSamples,2))

    for i in range(numSamples):
        answers[i][int(array[i])] = 1

    return data,answers

def main(args):

    accel_data = open("accel_data.txt", "r")
    accel_ans = open("accel_answer.txt", "r")

    test_data = open("test_data.txt","r")
    test_ans = open("test_ans.txt","r")

    test_data,test_ans = convertTextToData(test_data,test_ans)

    data,answers = convertTextToData(accel_data,accel_ans)
    
    model(data,answers,test_train=test_data,test_ans=test_ans)
    
   
if __name__ == "__main__":
    tf.app.run()


