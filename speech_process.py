# Python3
"""
@author : Prateeth Nayak
"""
from deepspeech import Model
from scipy.io import wavfile
import os

BEAM_WIDTH = 1024
LM_WEIGHT = 0.75
VALID_WORD_COUNT_WEIGHT = 1.85
N_FEATURES = 26
N_CONTEXT = 9
MODEL_FILE = 'pretrained/models/output_graph.pb'
ALPHABET_FILE = 'pretrained/models/alphabet.txt'
LANGUAGE_MODEL = 'pretrained/models/lm.binary'
TRIE_FILE = 'pretrained/models/trie'


def get_text(wav_file):

    curr_path = os.getcwd()
    print(MODEL_FILE)
    if os.path.isfile(MODEL_FILE):
        ds = Model(MODEL_FILE, N_FEATURES, N_CONTEXT, ALPHABET_FILE, BEAM_WIDTH)
        ds.enableDecoderWithLM(ALPHABET_FILE, LANGUAGE_MODEL, TRIE_FILE, LM_WEIGHT, VALID_WORD_COUNT_WEIGHT)
        fs, audio = wavfile.read(wav_file)
        processed_data = ds.stt(audio, fs)

        print(processed_data)


if __name__ == '__main__':
    get_text('file_example_WAV_2MG.wav')
