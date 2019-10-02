#Python3
"""
@author:Prateeth Nayak
"""


def create_pad_mask(x):
    mask = tf.cast(tf.math.equal(x,0), tf.float32)
    # (b, 1, 1, seq_len)
    return mask[:, tf.newaxis, tf.newaxis,:]

def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    pad_mask = create_pad_mask(x)
    return tf.maximum(look_ahead_mask, pad_mask)
