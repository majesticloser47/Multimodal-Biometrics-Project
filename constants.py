GENUINE = "Genuine"
FORGED = "Forged"
TRAIN = "Train"
TEST = "Test"
VALIDATION = "Val"
ALL = "All"
IDENTIFY = "Identify"
VERIFY = "Verify"
MAT_DATA_HEADERS = ['SubjectID',
 'Gen_Forge',
 'EEG',
 'ICA_EEG',
 'Diff',
 'SignWacom',
 'EEGHeader',
 'WacomHeader',
 'Note',
 'Path1',
 'Path2',
 'SignerID',
 'EpochLength']
epoch_frames = 1280 #frames
epoch_length = 10 #seconds
number_of_sections = 5
fps = epoch_frames / epoch_length
roi_starting_point = number_of_sections * epoch_frames
roi_idx = [roi_starting_point - 1, -1]
eeg_seq_len = 64
sign_seq_len = 256
eeg_samp_freq = 128