from pypianoroll import parse, Multitrack
from midi_io_musegan import findall_endswith, make_sure_path_exists
import os
from parameters import CONFIG
import numpy as np
from collections import defaultdict
import json
def converter(filepath,
              merge=True, # return (n_time_stamp, 128) or return (n_time_stamp, 128, num_track)
              velocity=False, # wether of not return the velocity in the value.(other wise return binary (0/1))
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
Get all the chord for the training set.
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
        index = np.argwhere(chord)
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
            result = converter(midi_path,merge=True, velocity=False)
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
            result = converter(midi_path, merge=False, velocity=False)
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

def get_nn_input(pianoroll_data, x_seq_length, y_seq_length, dictionary_dict):
    x = []
    y = []
    for i, piano_roll in enumerate(pianoroll_data):
        pos = 0
        x_tmp = []
        y_tmp = []
        x_list = [[] * piano_roll.shape[0]]
        index = np.argwhere(piano_roll)
        for row in index:
            x_list[row[0]].append(row[1])
        id_ = [dictionary_dict[str(v)] for v in x_list]

        while pos + x_seq_length + y_seq_length < piano_roll.shape[0]:
            x_tmp.append(id_[pos:pos + x_seq_length])
            y_tmp.append(id_[pos + x_seq_length: pos + x_seq_length + y_seq_length])
            pos += x_seq_length

        x_tmp = np.stack(x_tmp, axis=1)
        y_tmp = np.stack(y_tmp, axis=1)
        x.append(x_tmp)
        y.append(y_tmp)

    print(len(x))
    print("x shape", x[0].shape)
    print("y shape", y[0].shape)
    return x, y

if __name__ == "__main__":
    root_path = "../data/"
    ## test parser
    for midi_path in findall_endswith('.mid', root_path):
        result = converter(midi_path, merge=False, velocity=True)
    ## test get_dictionary_of_chord
    # get_dictionary_of_chord(root_path, two_hand=False)

    # test nn_input_generator
    with open("../data/output/chord_dictionary/two-hand.json", "r") as f:
        dictionary = json.load(f)

    get_nn_input





