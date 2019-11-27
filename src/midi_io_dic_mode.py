from pypianoroll import parse, Multitrack, Track
from midi_io_musegan import findall_endswith, make_sure_path_exists
import os
from parameters import CONFIG
import numpy as np
from collections import defaultdict
import json

"""
Use a new package (pypianoroll to process data)

Input one file path, output a pianoroll representation of the music.
If merge == True, then return (n_time_stamp, 128) other wish return (n_time_stamp, 128, num_track)
If velocity == True then return the velocity in the value divided by the maximum velocity value. (127)
(other wise return binary (0/1))
"""
def midiToPianoroll(filepath,
              merge=True,
              velocity=False,
              ):
    """Convert a MIDI file to a multi-track piano-roll and save the
    resulting multi-track piano-roll to the destination directory. Return a
    tuple of `midi_md5` and useful information extracted from the MIDI file.
    """
    midi_md5 = os.path.splitext(os.path.basename(filepath))[0]
    multitrack = Multitrack(filepath, beat_resolution=CONFIG['beat_resolution'], name=midi_md5)
    if merge:
        result = multitrack.get_merged_pianoroll(mode="max")
    else:
        result = multitrack.get_stacked_pianoroll()

    if not velocity:
        result = np.where(result > 0, 1, 0)
    else:
        result = result / CONFIG['velocity_high']
    return result

"""
pianoroll_data: List. One element corresponding to one piece of music
Do not support multi-track.
"""
def createSeqNetInputs(pianoroll_data: list, x_seq_length: int, y_seq_length: int) -> list:
    x = []
    y = []

    for i, piano_roll in enumerate(pianoroll_data):
        pos = 0
        x_tmp = []
        y_tmp = []
        while pos + x_seq_length + y_seq_length < piano_roll.shape[0]:
            x_tmp.append(piano_roll[pos:pos + x_seq_length])
            y_tmp.append(piano_roll[pos + x_seq_length: pos + x_seq_length + y_seq_length])
            pos += x_seq_length

        x_tmp = np.stack(x_tmp, axis=1)
        y_tmp = np.stack(y_tmp, axis=1)
        x.append(x_tmp)
        y.append(y_tmp)

    print(len(x))
    print("x shape", x[0].shape)
    print("y shape", y[0].shape)
    return x, y

"""
Get all the chords for the training set.
"""
def get_dictionary_of_chord(root_path,
                            two_hand=True # True if the chord is chord is played by two hand.
                            ):
    dir = "../output/chord_dictionary/"
    make_sure_path_exists(dir)

    def func(result):
        chord_set = set()
        chord_list = list()
        chord = np.unique(result, axis=0)
        index = np.argwhere(chord==1)
        _chord_tmp_dict = defaultdict(list)
        for row in index:
            _chord_tmp_dict[row[0]].append(row[1])

        chord_tmp_set = {str(value) for _, value in _chord_tmp_dict.items()}
        for i in chord_tmp_set:
            if i not in chord_set:
                chord_list.append(i)
                chord_set.add(i)

        return chord_list

    if two_hand:
        dic = dict()
        count = 0
        for midi_path in findall_endswith('.mid', root_path):
            result = midiToPianoroll(midi_path,merge=True, velocity=False)
            lis = func(result)
            for i in lis:
                if i not in dic.keys():
                    dic[i] = count
                    count += 1
        print(f"In total, there are {count} chords")
        with open(os.path.join(dir, "two-hand.json"), "w") as f:
            f.write(json.dumps(dic))
    else:
        dic_left = dict()
        dic_right = dict()
        count_left, count_right = 0, 0
        for midi_path in findall_endswith('.mid', root_path):
            result = midiToPianoroll(midi_path, merge=False, velocity=False)
            left = result[:, :, 1]
            right = result[:, :, 0]
            lis_left = func(left)
            lis_right = func(right)
            for i in lis_left:
                if i not in dic_left.keys():
                    dic_left[i] = count_left
                    count_left += 1

            for i in lis_right:
                if i not in dic_right.keys():
                    dic_right[i] = count_right
                    count_right += 1
        print(f"In total, there are {count_left}/{count_right} left/right-hand chords")

        with open(os.path.join(dir, "left-hand.json"), "w") as f:
            f.write(json.dumps(dic_left))

        with open(os.path.join(dir, "right-hand.json"), "w") as f:
            f.write(json.dumps(dic_right))


"""
Get NN input for the dictionary version.
pianoroll_data: List. One element corresponding to one piece of music
Do not support multi-track.
"""
def get_nn_input(pianoroll_data, x_seq_length, y_seq_length, dictionary_dict):
    x = []
    y = []
    for i, piano_roll in enumerate(pianoroll_data):
        pos = 0
        x_tmp = []
        y_tmp = []
        x_dict = defaultdict(list)
        index = np.argwhere(piano_roll==1)
        for row in index:
            x_dict[row[0]].append(row[1])
        id_ = [dictionary_dict[str(v)] for v in x_dict.values()]

        while pos + x_seq_length + y_seq_length < len(id_):
            x_tmp.append(id_[pos:pos + x_seq_length])
            y_tmp.append(id_[pos + x_seq_length: pos + x_seq_length + y_seq_length])
            pos += x_seq_length

        x.append(x_tmp)
        y.append(y_tmp)

    print(len(x))
    print(f"number of music: {len(x)}, number of samples = {len(x[0])}, dimension = {len(x[0][0])}")
    print("one sample of x", x[0][300])
    print("one sample of y", y[0][300])
    print(f"This input needs embedding or {len(dictionary_dict)} dims")

    return x, y


"""
Write the pianoroll output to midi file. 
Note: Do not support multi-track yet.
Input Shape: (n_time_stamp, 128)
"""
def pianorollToMidi(piano_roll: np.array,
                    name="test_midi",
                    dir="../output/",
                    velocity=False  # True if the input array contains velocity info (means not binary but continuous)
                    ):
    if velocity:
        piano_roll = np.floor(piano_roll * CONFIG['velocity_high'])
        piano_roll = np.clip(pianoroll_data, a_min=CONFIG['velocity_low'], a_max=CONFIG['velocity_high'])
    else:
        # fill in the default velocity
        piano_roll = np.where(piano_roll == 1, CONFIG['velocity'], 0)
        
    make_sure_path_exists(dir)
    track = Track(piano_roll, is_drum=False, name="piano")
    multi_track = Multitrack(tracks=[track],
                             tempo=CONFIG['tempo'],
                             downbeat=None,
                             beat_resolution=CONFIG['beat_resolution'],
                             )
    file_name = os.path.join(dir, name if name.endswith(".mid") else name+".mid")
    multi_track.write(file_name)

def load_corpus(path):
    with open(path, "r") as f:
        dic = json.load(f)
    return dic, len(dic)
if __name__ == "__main__":
    root_path = "../data/"
    ## 1. test parser
    # for midi_path in findall_endswith('.mid', root_path):
    #     result = midiToPianoroll(midi_path, merge=False, velocity=True)

    ## 2. test get_dictionary_of_chord
    #get_dictionary_of_chord(root_pat # h, two_hand=False)
    midi_path = next(findall_endswith('.mid', root_path))
    pianoroll_data = midiToPianoroll(midi_path, merge=True, velocity=False)

    ## 3. test pianoroll to midi file
    pianorollToMidi(pianoroll_data, name="test_midi.mid")

    ## 4. test createSeqNetInputs
    # createSeqNetInputs([pianoroll_data], 5, 5)


    ## 5. test nn_input_generator
    # with open("../output/chord_dictionary/two-hand.json", "r") as f:
    #     dictionary = json.load(f)
    #
    # x, y  =get_nn_input([pianoroll_data], 5, 5, dictionary)








