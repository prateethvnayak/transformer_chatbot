#Python3
"""
@author : Prateeth Nayak
"""
import tensorflow as tf
import preprocessing

BATCH_SIZE = 32
BUFFER_SIZE = 20000



def create_tf_data():
    questions, answers = preprocessing.prepare_data()
    dataset = tf.data.Dataset.from_tensor_slices((
                  {
                  'inputs':questions,
                  'dec_inputs':answers[:,:-1]
                  },
                  {
                  'outputs':answers[:,1:]
                  }
             ))
    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    print("\n\n\n",dataset)

if __name__ == '__main__':
    create_tf_data()
