
hidden_dim = 256
input_dim = 128
output_dim = 128
path = "../data/chp_op18.mid"
batch_size = 32
CONFIG = {
    'beat_resolution': 8, # temporal resolution (in time step per beat)
    'time_signatures': ['3/4'], # '3/4', '2/4'
    "velocity_high": 127,
    "velocity_low": 0,
    "tempo": 120.0, # default output tempo
    "velocity": 65 # default output velocity
}
threshold = 0
STAMPS_PER_BAR = CONFIG['beat_resolution'] * 3
input_num_bar = 2
output_num_bar = 2
time_len = STAMPS_PER_BAR * input_num_bar
output_len = STAMPS_PER_BAR * input_num_bar

END_TOKEN = 0
SILENCE_TOEKN = 1
emb_size = 256  # embedding size
root_path = "../data/"
non_trainable = False

