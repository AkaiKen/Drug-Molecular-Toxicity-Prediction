import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization(1)
        self.conv0_layer = tf.keras.layers.Conv2D(16, 7, 1, 'same', activation=tf.nn.relu)
        self.bn0 = tf.keras.layers.BatchNormalization(1)
        self.conv1_layer = tf.keras.layers.Conv2D(32, 5, 1, 'same', activation=tf.nn.relu)
        self.bn1 = tf.keras.layers.BatchNormalization(1)
        self.pool1_layer = tf.keras.layers.MaxPool2D(2, 2)
        self.do1 = tf.keras.layers.Dropout(0.3)
        self.conv2_layer = tf.keras.layers.Conv2D(64, 3, (1, 2), 'same', activation=tf.nn.relu)
        self.bn2 = tf.keras.layers.BatchNormalization(1)
        self.pool2_layer = tf.keras.layers.MaxPool2D(2, 2)
        self.do2 = tf.keras.layers.Dropout(0.3)
        # flat
        self.FCN1 = tf.keras.layers.Dense(364, activation=tf.nn.relu)
        # self.bn3 = tf.keras.layers.BatchNormalization(1)
        self.do3 = tf.keras.layers.Dropout(0.2)
        self.FCN2 = tf.keras.layers.Dense(2, activation=tf.nn.relu)
        # softmax

    def call(self, inputs):     
        x = inputs
        x = self.bn(x)
        x = self.conv0_layer(x)
        x = self.bn0(x)
        x = self.conv1_layer(x)
        x = self.bn1(x)
        x = self.pool1_layer(x)
        x = self.do1(x)
        x = self.conv2_layer(x)
        x = self.bn2(x)
        x = self.pool2_layer(x)
        x = self.do2(x)
        flat = tf.reshape(x, [-1, 18*50*64])
        flat = self.FCN1(flat)
        flat = self.do3(flat)
       #  flat = self.bn3(flat)
        output = self.FCN2(flat)
        output_with_sm = tf.nn.softmax(output)
        return output, output_with_sm
