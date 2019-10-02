#Python3
"""
@author : Prateeth Nayak
"""
import tensorflow as tf
from model_subcomponents import Multi_Head_Attn, positional_encoding

""" Encoder Layer """
def encoder_layer(units, model, num_heads, dropout, name='encode_layer'):

    inputs = tf.keras.Input(shape=(None, model), name='inputs')
    padding_mask = tf.keras.Input(shape=(1,1,None), name='padding_mask')

    attention = Multi_Head_Attn(
                model,
                num_heads,
                name='Attention')({
                'query':inputs,
                'key':inputs,
                'value':inputs,
                'mask':padding_mask
                })
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units,activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=model)(outputs)
    outputs = tf.keras.layers.Dense(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

""" Encoder """
def encoder(vocab_size, num_layers, units, model, num_heads, dropout, name='encoder'):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    pad_mask = tf.keras.Input(shape=(1,1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, model)(inputs)
    embeddings *= tf.ath.sqrt(tf.cast(model, tf.float32))
    embeddings = positional_encoding(vocab_size, model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=Dropout)(embeddings)

    for i in range(num_layers):
        outputs = encode_layer(units=units,
                               model=model,
                               num_heads=num_heads,
                               dropout=dropout,
                               name='encoder_layers_{}'.format(i))([outputs, padding_mask])
    return tf.keras.Model(inputs=[inputs, pad_mask], outputs=outputs, name=name)


""" Decoder Layer """
