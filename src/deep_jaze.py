from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np

''' Build a 2-layer LSTM from a training corpus '''
def build_model(corpus, val_indices, max_len, N_epochs=128):
    # number of different values or words in corpus
    N_values = len(set(corpus))

    # cut the corpus into semi-redundant sequences of max_len values
    step = 3
    sentences = []
    next_values = []
    for i in range(0, len(corpus) - max_len, step):
        sentences.append(corpus[i: i + max_len])
        next_values.append(corpus[i + max_len])
    print('nb sequences:', len(sentences))

    # transform data into binary matrices
    X = np.zeros((len(sentences), max_len, N_values), dtype=np.bool)
    y = np.zeros((len(sentences), N_values), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, val in enumerate(sentence):
            X[i, t, val_indices[val]] = 1
        y[i, val_indices[next_values[i]]] = 1

    # build a 2 stacked LSTM
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(max_len, N_values)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(N_values))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit(X, y, batch_size=128, nb_epoch=N_epochs)

    return model


''' Generates musical sequence based on the given data filename and settings.
    Plays then stores (MIDI file) the generated output. '''
def generate(data_fn, out_fn, N_epochs):
    # model settings
    max_len = 20
    max_tries = 1000
    diversity = 0.5

    # musical settings
    bpm = 130

    # get data
    chords, abstract_grammars = get_musical_data(data_fn)
    corpus, values, val_indices, indices_val = get_corpus_data(abstract_grammars)
    print('corpus length:', len(corpus))
    print('total # of values:', len(values))

    # build model
    model = lstm.build_model(corpus=corpus, val_indices=val_indices,
                             max_len=max_len, N_epochs=N_epochs)

    # set up audio stream
    out_stream = stream.Stream()

    # generation loop
    curr_offset = 0.0
    loopEnd = len(chords)
    for loopIndex in range(1, loopEnd):
        # get chords from file
        curr_chords = stream.Voice()
        for j in chords[loopIndex]:
            curr_chords.insert((j.offset % 4), j)

        # generate grammar
        curr_grammar = __generate_grammar(model=model, corpus=corpus,
                                          abstract_grammars=abstract_grammars,
                                          values=values, val_indices=val_indices,
                                          indices_val=indices_val,
                                          max_len=max_len, max_tries=max_tries,
                                          diversity=diversity)

        curr_grammar = curr_grammar.replace(' A',' C').replace(' X',' C')

        # Pruning #1: smoothing measure
        curr_grammar = prune_grammar(curr_grammar)

        # Get notes from grammar and chords
        curr_notes = unparse_grammar(curr_grammar, curr_chords)

        # Pruning #2: removing repeated and too close together notes
        curr_notes = prune_notes(curr_notes)

        # quality assurance: clean up notes
        curr_notes = clean_up_notes(curr_notes)

        # print # of notes in curr_notes
        print('After pruning: %s notes' % (len([i for i in curr_notes
            if isinstance(i, note.Note)])))

        # insert into the output stream
        for m in curr_notes:
            out_stream.insert(curr_offset + m.offset, m)
        for mc in curr_chords:
            out_stream.insert(curr_offset + mc.offset, mc)

        curr_offset += 4.0

    out_stream.insert(0.0, tempo.MetronomeMark(number=bpm))

    # Play the final stream through output (see 'play' lambda function above)
    play = lambda x: midi.realtime.StreamPlayer(x).play()
    play(out_stream)

    # save stream
    mf = midi.translate.streamToMidiFile(out_stream)
    mf.open(out_fn, 'wb')
    mf.write()
    mf.close()