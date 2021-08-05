import tensorflow as tf
import numpy as np



class attsets(tf.keras.layers.Layer):
    def __init__(self, num_slots = None, hidden_dim = 512, attention_output_length = 512):
        super(attsets, self).__init__()     
        
        self.dense1 = tf.keras.layers.Dense(256, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(256, activation = 'relu')
        self.dense3 = tf.keras.layers.Dense(256, activation = 'relu')
        self.dense4 = tf.keras.layers.Dense(attention_output_length, activation = tf.keras.layers.LeakyReLU())

        self.dense5 = tf.keras.layers.Dense(attention_output_length)
        self.softmax = tf.keras.layers.Softmax(1)

        self.dense6 = tf.keras.layers.Dense(attention_output_length)


    def call(self, inputs, embedded_pts, num_slots = None):

        inputs_init = tf.concat([inputs, embedded_pts ],-1)

        inputs = self.dense1(inputs_init)
        inputs = self.dense2(inputs)
        inputs = self.dense3(inputs)
        inputs = tf.concat([inputs, inputs_init], -1)
        inputs = self.dense4(inputs)

        mask = self.dense5(inputs)
        mask = tf.keras.layers.Softmax(1)(mask)
        att = inputs * mask
        output = tf.reduce_sum(att, 1)

        output = self.dense6(output)

        return output

if __name__ == "__main__":
    sample_input = tf.ones((20,12,57))
    embedded_pts = tf.ones((20,12,91))

    slot_att = slot_attention(256,512)
    print(slot_att(sample_input,embedded_pts).shape)
