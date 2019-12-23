# transformer_chatbot
Transformer based Conversational Bot

The implementation follows the paper [Attention is all you need](https://arxiv.org/abs/1706.03762)

The conversational has two-stage module:
[ ] - Speech-to-Text 
    [x] - Deep-Speech pre-trained model
    [ ] - Re-trained model for Common-Voice Dataset
[ ] - Text-to-Text Response module
    [ ] - Transformer model
       [ ] - preprocessing (word tokenizer + positional encoding)
       [ ] - 
`wget https://github.com/mozilla/DeepSpeech/releases/download/v0.5.1/deepspeech-0.5.1-models.tar.gz`

`ffmpeg -i 111.mp3 -acodec pcm_s16le -ac 1 -ar 16000 out.wav`

`http://www.manythings.org/audio/sentences/`
