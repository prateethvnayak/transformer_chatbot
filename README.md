# transformer_chatbot
Transformer based Conversational Bot

The implementation follows the paper [Attention is all you need](https://arxiv.org/abs/1706.03762)

The conversational has two-stage module:
- [ ] Speech-to-Text 
    - [x] Deep-Speech pre-trained model
    - [ ] Re-trained model for Common-Voice Dataset
- [ ] Text-to-Text Response module
    - [x] Transformer model
        - [x] preprocessing (word tokenizer + positional encoding)
        - [x] Encoder (N = 6) Layers
        - [x] Decoder 
        - [x] Adam with custom Lr Schedule
    - [x] Trained on Cornell Movie Dialog Corpus
    - [ ] Analysis of Lr Schedule based on Paper

Speech-to-Text Deep Module References:

`wget https://github.com/mozilla/DeepSpeech/releases/download/v0.5.1/deepspeech-0.5.1-models.tar.gz`

Converting Sample Audio Samples for 16-bit, 16Hz, Mono-Channel WAV-Audio:
`ffmpeg -i 111.mp3 -acodec pcm_s16le -ac 1 -ar 16000 out.wav`

Ref for Dataset and Sample Audio:
- [Cornell Movie Dialog Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
- [Common Voice Dataset](https://voice.mozilla.org/en/datasets)
- [Sample Audio Samples WAV](http://www.manythings.org/audio/sentences/)
