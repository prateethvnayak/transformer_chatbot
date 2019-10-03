# Python3
"""
@author: Prateeth Nayak
"""
import tensorflow as tf
# INDIVIDUAL MODEL COMPONENTS


def scaled_dotprod_attention(Q, K, V, mask):
    # Calculates the attention weights with dot-product of softmx and value vector
    qk = tf.matmul(Q, K, transpose_b=True)

    depth = tf.cast(tf.shape(K)[-1], tf.float32)
    logits = qk / tf.math.sqrt(depth)

    # adding mask to the remove padded tokens
    if mask is not None:
        logits += (mask * -1e9)

    attention_w = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_w, V)

    return output


class Multi_Head_Attn(tf.keras.layers.Layer):

    def __init__(self, model, num_heads, name='multi_head_attention'):
        super(Multi_Head_Attn, self).__init__(name=name)
        self.num_head = num_heads
        self.model = model
        assert model % num_heads == 0
        self.depth = model // self.num_head

        self.Q_dense = tf.keras.layers.Dense(units=model)
        self.K_dense = tf.keras.layers.Dense(units=model)
        self.V_dense = tf.keras.layers.Dense(units=model)

        self.dense = tf.keras.layers.Dense(units=model)

    def _split_heads(self, inputs, bsize):
        inputs = tf.reshape(inputs, shape=(bsize, -1, self.num_head, self.depth))

        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        Q, K, V, mask = inputs['query'], inputs['key'], inputs['value'], \
            inputs['mask']
        b_size = tf.shape(Q)[0]

        # linear
        Q = self.Q_dense(Q)
        K = self.K_dense(K)
        V = self.V_dense(V)

        # split heads
        Q = self._split_heads(Q, b_size)
        K = self._split_heads(K, b_size)
        V = self._split_heads(V, b_size)

        # scaled dot prod attention
        scaled_attn = scaled_dotprod_attention(Q, K, V, mask)
        scaled_attn = tf.transpose(scaled_attn, perm=[0, 2, 1, 3])

        # concat
        concat_attn = tf.reshape(scaled_attn,
                                 (b_size, -1, self.model))
        # linear
        outputs = self.dense(concat_attn)
        return outputs


class positional_encoding(tf.keras.layers.Layer):

    def __init__(self, position, model):
        super(positional_encoding, self).__init__()
        self.pos_encode = self.pos_encoding(position, model)

    def get_angles(self, position, i, model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(model, tf.float32))
        return position * angles

    def pos_encoding(self, position, model):
        angle_r = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(model, dtype=tf.float32)[tf.newaxis, :],
            model=model)
        sin_val = tf.math.sin(angle_r[:, 0::2])
        cos_val = tf.math.cos(angle_r[:, 1::2])

        pos_encode = tf.concat([sin_val, cos_val], axis=-1)
        pos_encode = pos_encode[tf.newaxis, ...]
        return tf.cast(pos_encode, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encode[:, :tf.shape(inputs)[1], :]
