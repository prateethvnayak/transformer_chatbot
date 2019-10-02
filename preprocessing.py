#python3
"""
@author :Prateeth Nayak
"""
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import random
import re

_DATA_PATH = '../cornell_movie_dialogs'
path_movie_lines = os.path.join(_DATA_PATH,'movie_lines.txt')
path_movie_conv = os.path.join(_DATA_PATH, 'movie_conversations.txt')

MAX_LENGTH = 40
MAX_SAMPLES = 25000

# UTILS
def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # "He is here." => "He is here ."
    sentence = re.sub(r"([?.!,])", r" \1 ",sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()

    return sentence

def tokenize(inputs, outputs, tkenizer, strt_tk, end_tk):
    token_in, token_out = [], []
    for (sentence1, sentence2) in zip(inputs, outputs):
        sentence1 = strt_tk + tkenizer.encode(sentence1) + end_tk
        sentence2 = strt_tk + tkenizer.encode(sentence2) + end_tk
        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
            token_in.append(sentence1)
            token_out.append(sentence2)
    #pad tokenized sentences
    token_in = tf.keras.preprocessing.sequence.pad_sequences(
                token_in, maxlen=MAX_LENGTH,padding='post')
    token_out = tf.keras.preprocessing.sequence.pad_sequences(
                token_out, maxlen=MAX_LENGTH, padding='post')

    return token_in, token_out


def load_data_conv():
    id_2_line = {}

    with open(path_movie_lines,errors='ignore') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.replace('\n','').split(' +++$+++ ')
        id_2_line[parts[0]] = parts[4]

    inputs, outputs  = [], []
    with open(path_movie_conv, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.replace('\n','').split(' +++$+++ ')

        conv = [line[1:-1] for line in parts[3][1:-1].split(', ')]
        for i in range(len(conv)-1):
            inputs.append(preprocess_sentence(id_2_line[conv[i]]))
            outputs.append(preprocess_sentence(id_2_line[conv[i+1]]))
            if len(outputs) >= MAX_SAMPLES:
                return inputs, outputs
    return inputs, outputs

# MAIN FUNC
def prepare_data():
    # Wrapper Function
    questions, answers = load_data_conv()
    # print Sample
    print("\n Sample Data Point:\n")
    print("Sample question :{}".format(questions[20]))
    print("Sample answer :{}".format(answers[20]))

    # Create tokenizers
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(\
    questions + answers, target_vocab_size=2**13)
    # start and end token to indicate start & end of sentence
    start_tk, end_tk = [tokenizer.vocab_size],[tokenizer.vocab_size + 1]
    #vocab size along with start and end token
    vocab_size = tokenizer.vocab_size + 2

    print("\n Tokenized sample question: {}".format(tokenizer.encode(questions[20])))

    questions, answers = tokenize(questions, answers,\
                                  tokenizer, start_tk, end_tk)
    print('Vocab Size :{}'.format(vocab_size))
    print('No. of Samples :{}\n\n\n'.format(len(questions)))


    return questions, answers


# if __name__ == '__main__':
#     prepare_data()
