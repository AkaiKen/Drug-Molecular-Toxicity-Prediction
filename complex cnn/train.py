import os
import numpy as np
import pandas as pd
import tensorflow as tf
from model import MyModel



def load_data(filefolder):
    data = np.load(os.path.abspath(filefolder + '/names_onehots.npy'), allow_pickle=True).item()
    data = data['onehots']
    label = pd.read_csv(os.path.abspath(filefolder + '/names_labels.txt'), sep=',')
    label = label['Label'].values
    return data, label

if __name__ == '__main__':
    # parameters
    LR = 0.002#2e-3
    BatchSize = 128
    EPOCH = 3

    # data
    train_x, train_y = load_data('../train')
    valid_x, valid_y = load_data('../validation')

    # model & input and output of model
    tf.reset_default_graph()
    model = MyModel()

    onehots_shape = list(train_x.shape[1:])
    input_place_holder = tf.placeholder(tf.float32, [None] + onehots_shape, name='input')
    input_place_holder_reshaped = tf.reshape(input_place_holder, [-1] + onehots_shape + [1])
    label_place_holder = tf.placeholder(tf.int32, [None], name='label')
    label_place_holder_2d = tf.one_hot(label_place_holder, 2)
    output, output_with_sm = model(input_place_holder_reshaped)
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
                tf.set_random_seed(i)
                b_x, b_y = train_x[i:i + BatchSize], train_y[i:i + BatchSize]
                _, loss_ = sess.run([train_op, loss], {'input:0': b_x, 'label:0': b_y})
                # print("Epoch {}: [{}/{}], training set loss: {:.4}".format(epoch, i, train_size, loss_))

            if epoch % 1 == 0:
                val_prediction = sess.run(output_with_sm, {'input:0': valid_x})
                val_prediction = val_prediction[:, 1]
                auc_value = sess.run(update_op, feed_dict={prediction_place_holder: val_prediction, label_place_holder: valid_y})
                print("auc_value", auc_value)
                if auc_value > best_val_auc:
                    best_val_auc = auc_value
                    print("update")
                    saver.save(sess, './weights/model')
