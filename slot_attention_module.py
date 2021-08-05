import tensorflow as tf
import numpy as np

#Gist taken from torch implementation by Lucidrains

class slot_attention(tf.keras.layers.Layer):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128, name = None, attention_output_length = 256):
        super(slot_attention, self).__init__(name=name)
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.dim = dim

        mu_init = tf.constant_initializer(value=tf.random.normal(shape = (1,1, dim)).numpy())
        sigma_init = tf.constant_initializer(value=tf.random.normal(shape = (1,1, dim)).numpy())

        self.slots_mu = self.add_weight(shape=(1, 1, dim), initializer=mu_init)
        self.slots_sigma = self.add_weight(shape=(1, 1, dim), initializer=sigma_init)

        self.to_q = tf.keras.layers.Dense(dim, use_bias = False)
        self.to_k = tf.keras.layers.Dense(dim, use_bias = False)
        self.to_v = tf.keras.layers.Dense(dim, use_bias = False)

        self.gru = tf.keras.layers.GRUCell(dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp1 = tf.keras.layers.Dense(hidden_dim, activation = 'relu')
        self.mlp2 = tf.keras.layers.Dense(dim)

        self.norm_input  = tf.keras.layers.LayerNormalization(-1)
        self.norm_slots  = tf.keras.layers.LayerNormalization(-1)
        self.norm_pre_ff = tf.keras.layers.LayerNormalization(-1)

        self.dense1 = tf.keras.layers.Dense(64, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(32, activation = 'relu')
        self.dense3 = tf.keras.layers.Dense(32, activation = 'relu')
        self.dense4 = tf.keras.layers.Dense(32, activation = 'relu')
        self.dense5 = tf.keras.layers.Dense(32, activation = 'relu')
        self.dense6 = tf.keras.layers.Dense(32)

        self.dense7 = tf.keras.layers.Dense(dim, activation = tf.keras.layers.LeakyReLU())


    def call(self, inputs, embedded_pts, num_slots = None):


        max_rel = self.dense1(inputs)
        max_rel = self.dense2(max_rel)
        max_rel = self.dense3(max_rel)
        max_rel = self.dense4(max_rel)
        max_rel = self.dense5(max_rel)
        max_rel = self.dense6(max_rel)


        inputs = tf.concat([inputs, embedded_pts, max_rel ],-1)
        inputs = self.dense7(inputs)



        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = tf.broadcast_to(self.slots_mu, (b, n_s, self.dim))
        sigma = tf.broadcast_to(self.slots_sigma, (b, n_s, self.dim))
        slots = tf.random.normal(mu.shape, mean = mu, stddev = sigma)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = tf.einsum('bid,bjd->bij', q, k) * self.scale
            attn = tf.keras.layers.Softmax(1)(dots) + self.eps
            attn = attn / tf.math.reduce_sum(attn, axis = -1, keepdims = True)

            updates = tf.einsum('bjd,bij->bid', v, attn)

            _,[slots] = self.gru(inputs = tf.reshape(updates, (-1, d)), states = [tf.reshape(slots_prev, (-1, d))] )

            slots = tf.reshape(slots, (b, -1, d))
            mlp_result = self.mlp2(self.mlp1(self.norm_pre_ff(slots)))
            slots = slots + mlp_result

        ### This isn't part of the original slot attention model, but downsamples result if necessary to correct length
        attention_outputs = tf.reshape(slots, [-1, n_s * self.dim])

        return attention_outputs

if __name__ == "__main__":
    sample_input = tf.ones((20,12,57))
    indices = tf.ones((20,12,2))
    embedded_pts = tf.ones((20,12,91))

    slot_att = slot_attention(4,256)
    slot_att(sample_input,embedded_pts,indices)


    print(len(slot_att.trainable_weights))
