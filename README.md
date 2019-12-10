# MIDI-Based Music Generation
*This repo contains the model we used to generate MIDI music, each branch is a model we used, the list below briefly illustrate each model. You could switch to any branch use "git checkout" insturction. More detailed information could be found in our report.*

* __dictionary__: A simple embedding-based LSTM implementation.

*This architecture above refers to the Simple LSTM Model in our report.*

* __master__: A simple Pianoroll-based encoder and decoder implementation and MIDI I/O.
* __baseline_gpu__: A simple Pianoroll-based encoder and decoder implementation identical to master branch but in gpu version.
* __dict_AE__: An embedding-based encoder and decoder implementation.

*These three architecture above refer to the Simple LSTM Encoder-Decoder Model in our report.*

* __dict_AE_attention__: A embedding-based encoder and decoder implementation with attention structure.
* __dict_attention_left_right__: A embedding-based Dual-track encoder and decoder implementation with attention structure.
* __left_right__: A Pianoroll-based Dual-track encoder and decoder implementation with attention structure.

*These three architecture above refer to the Attention-Based-LSTM Encoder-Decoder Model in our report.*

* __left_right_CNN__: A Pianoroll-based Dual-track encoder and decoder implementation with attention structure, but also have two 1d convolutional layers before the encoder, this one refers to the CNN+Attention-Based-LSTM Encoder-Decoder Model in our report.

* __velocity__: A simple Pianoroll-based encoder and decoder implementation focusing on predicting the velocity of each chord, no show in our report for bad performance.
