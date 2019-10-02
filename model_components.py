#Python3
"""
@author: Prateeth Nayak
"""

# INDIVIDUAL MODEL COMPONENTS
def scaled_dotprod_attention(Q, K, V, mask):
    # Calculates the attention weights with dot-product of softmx and value vector
    qk = tf.matmul(Q, K), transpose_b=True)

    depth = tf.cast(tf.shape(K)[-1],tf.float32)
    logits = qk / tf.math.sqrt(depth)

    # adding mask to the remove padded tokens
    if mask is not None:
        logits += (mask * -1e9)

    attention_w = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_w, V)

    return output

class Multi_Head_Attn(tf.keras.layers.layer):

    def __init__(self, model, num_heads, name = 'multi_head_attention'):
        super(Multi_Head_Attn, self).__init__(name=name)
        self.no_head = num_heads
        self.model = model
        assert model % num_heads == 0
        self.depth = model // self.no_head

        self.Q_dense = tf.keras.layers.Dense(units=model)
        self.K_dense = tf.keras.layers.Dense(units=model)
        self.V_dense = tf.keras.layers.Dense(units=model)

        self.dense = tf.keras.layers.Dense(units=model)

    def _split_heads(self, inputs, bsize):
        inputs = tf.reshape(inputs,shape=(bsize, -1, self.num_heads, self.depth))

        return tf.transpose(inputs, perm=[0,2,1,3])

    def call(self, inputs):
        Q, K, V, mask = inputs['query'], inputs['key'], inputs['value'], \
                        inputs['mask']
        b_size = tf.shape(Q)[0]

        # linear
        Q = self.Q_Dense(Q)
        K = self.K_Dense(K)
        V = self.V_Dense(V)

        # split heads
        Q = self._split_heads(Q,b_size)
        K = self._split_heads(K,b_size)
        V = self._split_heads(V,b_size)

        # scaled dot prod attention
        scaled_attn = scaled_dotprod_attention(Q, K, V, mask)
        scaled_attn = tf.transpose(scaled_attn, perm=[0,2,1,3])

        #concat
        concat_attn = tf.reshape(scaled_attn,\
                      (b_size, -1, self.model))
        #linear
        outputs = self.dense(concat_attn)
        return outputs
