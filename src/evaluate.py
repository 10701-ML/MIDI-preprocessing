from midi_io_dic_mode import *
from parameters import STAMPS_PER_BAR
import numpy as np
def evaluate(midi_path):
    pianoroll = midiToPianoroll(midi_path)  # (time, 128)
    # UPC:number of used pitch classes per bar (from 0 to 12).
    barsize = STAMPS_PER_BAR
    size = pianoroll.shape[0]
    upcs = []
    for i in range(0, size, barsize):
        sub_piano = pianoroll[i: i+barsize, :]
        upc_ = np.sum(np.max(sub_piano, axis=0))
        upcs.append(upc_)

    upc = sum(upcs) / len(upcs)
    # QN: ratio of “qualified” notes( in % ).We
    qn = None
    valid_key_num = 0
    total_key_num = 0
    for note in range(128):
        note_piano = pianoroll[:, note] # (time,)
        key_on = False
        key_len = 0
        for note_status in note_piano:
            if not key_on and note_status > 0:
                key_on = True
                key_len = 1

            elif key_on and note_status > 0:
                key_len += 1

            elif note_status <= 0 and key_on:
                key_on = False
                total_key_num += 1
                valid_key_num = valid_key_num if key_len < 3 else valid_key_num+1
                key_len = 0
            else:
                pass

    qn = valid_key_num / total_key_num
    return upc, qn


if __name__ == "__main__":
    midi_path = "../data/chp_op18.mid"
    evaluate(midi_path)