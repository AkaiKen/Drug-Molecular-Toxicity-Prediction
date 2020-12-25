import os
import tensorflow as tf
import numpy as np
import pandas as pd
import random
BatchSize = 50

def load_test_data_name(filefolder):
    data = np.load(os.path.abspath(filefolder + '/names_onehots.npy'), allow_pickle=True).item()
    onehots = data['onehots']
    name = data['names']
    return onehots, name

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1_layer = tf.keras.layers.Conv2D(32, 5, 1, 'same', activation=tf.nn.relu)
        self.pool1_layer = tf.keras.layers.MaxPool2D(2, 2)
        self.conv2_layer = tf.keras.layers.Conv2D(32, 3, (1, 2), 'same', activation=tf.nn.relu)
        self.pool2_layer = tf.keras.layers.MaxPool2D(2, 2)
        # flat
        self.FCN = tf.keras.layers.Dense(2)
        # softmax

    def call(self, inputs):
        x = self.conv1_layer(inputs)
        x = self.pool1_layer(x)
        x = self.conv2_layer(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.nn.leaky_relu(x)
        x = self.pool2_layer(x)
        x = tf.layers.flatten(x)
        x = tf.expand_dims(x, dim=1)
        print("INPUT pool1----------",x.shape)
        
       # print("INPUT conv2----------",x.shape)
       # x = self.pool2_layer(x)
       # print("INPUT pool2----------",x.shape)
        lstmCell = tf.keras.layers.LSTM(512)(x)
        print("INPUT lstm----------",lstmCell.shape)
        x = tf.layers.flatten(lstmCell)
        x = tf.keras.layers.Dense(256)(x)
        output = self.FCN(x)
        output_with_sm = tf.nn.softmax(output)
        return output, output_with_sm
    

if __name__ == '__main__':
    # data
    test_data, test_name = load_test_data_name("../test")
    name = test_name

    # model
    tf.reset_default_graph()
    model = MyModel()
    input_place_holder = tf.placeholder(tf.float32, [None] + list(test_data.shape[1:]), name='input')
    input_place_holder_reshaped = tf.reshape(input_place_holder, [-1] + list(test_data.shape[1:]) + [1])
    output, output_with_sm = model(input_place_holder_reshaped)

    # Predict on the test set
    data_size = test_data.shape[0]
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, os.path.abspath('./weights/model'))
        prediction = []
        for i in range(0, data_size, BatchSize):
            print(i)
            test_output = sess.run(output, {'input:0': test_data[i:i + BatchSize]})
            test_output_with_sm = sess.run(output_with_sm, {'input:0': test_data[i:i + BatchSize]})
            pred = test_output_with_sm[:, 1]
            prediction.extend(list(pred))
    sess.close()
    f = open('output_518030990028.txt', 'w')
    f.write('Chemical,Label\n')
    for i, v in enumerate(prediction):
        f.write(name[i] + ',%f\n' % v)
    f.close()