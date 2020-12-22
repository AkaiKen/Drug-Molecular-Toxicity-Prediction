import os
import numpy as np
import pandas as pd
import tensorflow as tf

def load_data(filefolder):
    data = np.load(os.path.abspath(filefolder + '/names_onehots.npy'), allow_pickle=True).item()
    data = data['onehots']
    label = pd.read_csv(os.path.abspath(filefolder + '/names_labels.txt'), sep=',')
    label = label['Label'].values
    return data, label


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
        print("INPUT input----------",inputs.shape)
        x = self.conv1_layer(inputs)
        print("INPUT conv1----------",x.shape)
        x = self.pool1_layer(x)
        print("INPUT pool1----------",x.shape)
        x = self.conv2_layer(x)
        print("INPUT conv2----------",x.shape)
        x = self.pool2_layer(x)
        print("INPUT pool2----------",x.shape)
        flat = tf.reshape(x, [-1, 18*50*32])
        output = self.FCN(flat)
        output_with_sm = tf.nn.softmax(output)
        return output, output_with_sm

if __name__ == '__main__':
    # parameters
    LR = 0.01
    BatchSize = 128
    EPOCH = 1

    # data
    train_x, train_y = load_data('../train')
    valid_x, valid_y = load_data('../validation')
    
    print("OUTPUT train_x----------",train_x.shape)

    # model & input and output of model
    tf.reset_default_graph()
    model = MyModel()

    onehots_shape = list(train_x.shape[1:])
    print("OUTPUT onehots----------",onehots_shape)
    input_place_holder = tf.placeholder(tf.float32, [None] + onehots_shape, name='input')
    print("OUTPUT holder1----------",input_place_holder.shape)
    input_place_holder_reshaped = tf.reshape(input_place_holder, [-1] + onehots_shape + [1])
    print("OUTPUT holder2----------",input_place_holder_reshaped.shape)
    label_place_holder = tf.placeholder(tf.int32, [None], name='label')
    print("OUTPUT holder3----------",label_place_holder.shape)
    label_place_holder_2d = tf.one_hot(label_place_holder, 2)
    print("OUTPUT holder4----------",label_place_holder_2d.shape)
    output, output_with_sm = model(input_place_holder_reshaped)
    print("OUTPUT 1----------",output.shape)
    print("OUTPUT 2----------",output_with_sm.shape)
    model.summary()  # show model's structure

    # loss
    bce = tf.keras.losses.BinaryCrossentropy()  # compute cost
    loss = bce(label_place_holder_2d, output_with_sm)

    # Optimizer
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)

    # auc
    prediction_place_holder = tf.placeholder(tf.float64, [None], name='pred')
    auc, update_op = tf.metrics.auc(labels=label_place_holder, predictions=prediction_place_holder)



    # run
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        saver = tf.train.Saver()

        train_size = train_x.shape[0]
        best_val_auc = 0
        for epoch in range(EPOCH):
            for i in range(0, train_size, BatchSize):
                b_x, b_y = train_x[i:i + BatchSize], train_y[i:i + BatchSize]
                _, loss_ = sess.run([train_op, loss], {'input:0': b_x, 'label:0': b_y})
                print("Epoch {}: [{}/{}], training set loss: {:.4}".format(epoch, i, train_size, loss_))

            if epoch % 1 == 0:
                val_prediction = sess.run(output_with_sm, {'input:0': valid_x})
                val_prediction = val_prediction[:, 1]
                auc_value = sess.run(update_op, feed_dict={prediction_place_holder: val_prediction, label_place_holder: valid_y})
                print("auc_value", auc_value)
                if auc_value > best_val_auc:
                    saver.save(sess, './weights/model')
