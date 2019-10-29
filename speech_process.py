# Python3
"""
@author : Prateeth Nayak
"""
import tensorflow as tf
from deepspeech import Model
from scipy.io import wavfile
import os
import pdb
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

BEAM_WIDTH = 500
LM_WEIGHT = 1.50
VALID_WORD_COUNT_WEIGHT = 2.25
N_FEATURES = 26
N_CONTEXT = 9
MODEL_FILE = 'pretrained/models/output_graph.pbmm'
ALPHABET_FILE = 'pretrained/models/alphabet.txt'
LANGUAGE_MODEL = 'pretrained/models/lm.binary'
TRIE_FILE = 'pretrained/models/trie'


def get_text(wav_file):

    # curr_path = os.getcwd()
    # print(MODEL_FILE)
    # if os.path.isfile(MODEL_FILE):
    ds = Model(MODEL_FILE, N_FEATURES, N_CONTEXT, ALPHABET_FILE, BEAM_WIDTH)
    ds.enableDecoderWithLM(ALPHABET_FILE, LANGUAGE_MODEL, TRIE_FILE, LM_WEIGHT, VALID_WORD_COUNT_WEIGHT)
    fs, audio = wavfile.read(wav_file)
    # pdb.set_trace()
    # audio = audio[:, 0]
    # audio = audio.reshape(-1, 1)
    processed_data = ds.stt(audio, fs)

    print(processed_data)


if __name__ == '__main__':
    get_text('../out3.wav')
