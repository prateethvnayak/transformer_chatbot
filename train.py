# Python3
"""
@author : Prateeth Nayak
"""
import tensorflow as tf
from model import transformer
from create_tf_dataset import create_data
import pdb
from preprocessing import preprocess_sentence
# tf.keras.backend.clear_session()
# HYPER PARAMS
NUM_LAYERS = 2
MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1
MAX_LENGTH = 40  # (same as in preprocessing.py)
EPOCHS = 20


def loss_func(y_true, y_pred):

    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


dataset, other_tuple = create_data()
model = transformer(
    vocab_size=other_tuple[0],
    num_layers=NUM_LAYERS,
    units=UNITS,
    model=MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

learning_rate = CustomSchedule(d_model=128)
opt = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


def accuracy(y_true, y_pred):
    # first make sure both have the same length (b_size, MAX_LENGTH -1)
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)




model.compile(optimizer=opt, loss=loss_func, metrics=[accuracy])
model.summary()
model.fit(dataset, epochs=20)
def predict(model, tokenizer, strt_tk, end_tk, sentence):
    sentence = preprocess_sentence(sentence)
    sentence = tf.expand_dims(strt_tk + tokenizer.encode(sentence) + end_tk, axis=0)
    output = tf.expand_dims(strt_tk, 0)
    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if tf.equal(predicted_id, end_tk[0]):
            break
        output = tf.concat([output, predicted_id], axis=-1)
    predicted_sentence = tf.squeeze(output, axis=0)
    inference_sentence = tokenizer.decode(
    [i for i in predicted_sentence if i < tokenizer.vocab_size])

    return inference_sentence

def evaluate(sentence):
    print("\nEvaluating...")
    print("Input Sentence : {}".format(sentence))
    output = predict(model, other_tuple[1], other_tuple[2], other_tuple[3], sentence)
    print("Output Sentence :{}".format(output))
pdb.set_trace()
