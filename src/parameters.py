time_len = 200
output_len = 200
hidden_dim = 256
input_dim = 88
output_dim = 88
path = "../data/chp_op18.mid"
threshold = 0
CONFIG = {
    'beat_resolution': 24, # temporal resolution (in time step per beat)
    'time_signatures': ['4/4'], # '3/4', '2/4'
    "velocity_high": 127,
    "velocity_low": 0,
    "tempo": 120.0, # default output tempo
    "velocity": 65 # default output velocity
}

emb_size = 256 # embedding size
root_path = "../data/"

