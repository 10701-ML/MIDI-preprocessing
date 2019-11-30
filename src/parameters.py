
hidden_dim = 256
input_dim = 88
output_dim = 88
path = "../data/chp_op18.mid"
CONFIG = {
    'beat_resolution': 24, # temporal resolution (in time step per beat)
    'time_signatures': ['4/4'], # '3/4', '2/4'
    "velocity_high": 127,
    "velocity_low": 0,
    "tempo": 120.0, # default output tempo
    "velocity": 65 # default output velocity
}
STAMPS_PER_BAR = CONFIG['beat_resolution'] * 4 # assuming 4/4.
input_num_bar = 4
output_num_bar = 4
time_len = STAMPS_PER_BAR * input_num_bar
output_len = STAMPS_PER_BAR * input_num_bar

END_TOKEN = 0
SILENCE_TOEKN = 1
emb_size = 128 # embedding size
root_path = "../data/"

