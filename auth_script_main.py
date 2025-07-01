import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import constants
import os
from dotenv import load_dotenv
import matplotlib.cm as cm
import scipy.io as sp
import json
import pprint as pp
from scipy.signal import welch
from scipy.integrate import trapezoid
import torch
from torch.nn.utils.rnn import pad_sequence

load_dotenv()
dataset_path = os.getenv('DATASET_PATH')

def get_dataset_files_and_user_ids(data_category = constants.GENUINE, data_type = constants.TRAIN):
    user_ids = []
    files_csv = []
    files_mat = []

    # Get training and testing data
    data_split = pd.read_csv(os.path.join(dataset_path, "Identification_split.csv"))
    training_data_files = data_split[data_split.set == constants.TRAIN].filename.str.rsplit('.', n=1).str[0]
    # print(training_data_files) # only for debugging

    # TODO: get file sbased on type of type required, i.e. Training, tetsing or validation

    for root, dirs, files in os.walk(dataset_path):
        if os.path.basename(root) == constants.GENUINE == data_category:
            for file in files:
                if file.endswith('.csv'):
                    files_csv.append(os.path.join(root, file))
                elif file.endswith('.mat'):
                    files_mat.append(os.path.join(root, file))
        elif os.path.basename(root) == constants.FORGED == data_category:
            for file in files:
                if file.endswith('.csv'):
                    files_csv.append(os.path.join(root, file))
                elif file.endswith('.mat'):
                    files_mat.append(os.path.join(root, file))
        if os.path.basename(root) != constants.GENUINE and os.path.basename(root) != constants.FORGED and os.path.basename(root) != 'SignEEGv1.0':
            user_ids.append(os.path.basename(root))
    files_csv = sorted(files_csv, key=lambda x: int(x.split('_')[3].split(".")[0]))
    files_mat = sorted(files_mat, key=lambda x: int(x.split('_')[3]))
    return files_csv, files_mat, user_ids

def get_user_csv_sign_data_cleaned(user_sign_data_csv): #Provide file name of the csv file
    content = pd.read_csv(user_sign_data_csv, skiprows=1, header=None)
    content.drop
    content.columns = [c.strip() for c in content.iloc[0]] #gettting rid of extra space in column names
    content = content.iloc[1:]
    return content

def normalize_sign_data(data):
    x = np.array(data['X']).astype(int)
    y = np.array(data['Y']).astype(int)
    t = np.array(data['T']).astype(int)
    pressure = np.array(data['Pressure']).astype(int)
    azimuth = np.array(data['Azimuth']).astype(int)
    altitude = np.array(data['Altitude']).astype(int)
    # normalize signature data
    norm_x = x / np.max(x)
    norm_y = y / np.max(y)
    norm_pressure = pressure / np.max(pressure)
    norm_azimuth = azimuth / np.max(azimuth)
    norm_altitude = altitude / np.max(altitude)
    return norm_x, norm_y, t, norm_pressure, norm_azimuth, norm_altitude

def get_user_mat_data(user_id=None):
    if user_id is None:
        user_id = user_ids[0]  # Default to the first user if none specified
    user_files = [x for x in mat_data if user_id in x]
    user_files_sorted = pd.Series(user_files)
    user_files_sorted.sort_values(key=lambda x: x.str.split('_').str[3].astype(int), inplace=True)
    user_files_reset = user_files_sorted.reset_index(drop=True)
    # print(user_files_reset)
    return user_files_reset

def normalize_eeg_data(eeg_input):
    norm_eeg_data = []  
    for channel in eeg_input:
        mean = np.mean(channel)
        std = np.std(channel)
        norm_channel = (channel - mean)/std
        norm_eeg_data.append(norm_channel)
    norm_eeg_array = np.array(norm_eeg_data)
    # print(norm_eeg_data)
    return norm_eeg_array

def get_signature_feature_vector(path, user_ids, seq_len = 256, overlap = 0.5):
    user_id = [id for id in user_ids if id in path][0]
    sign_data = get_user_csv_sign_data_cleaned(path)
    x, y, t, pressure, azimuth, altitude = normalize_sign_data(sign_data)

    # Calculate pen velocity
    dt = 1 / (4 / 1000)
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    v = np.sqrt(vx**2 + vy**2)
    
    # Calculate pen acceleration
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    a = np.sqrt(ax**2 + ay**2)

    # Calculate number of pen lifts
    # Do a logical & betwen the values of the array(except for the last) are > 0 and the values for which (except the first element) > 0
    pen_lifts = np.sum((pressure[:-1] > 0) & (pressure[1:] == 0))
    # print(pen_lifts)

    # Calculate stroke duration
    is_pen_down = pressure > 0 
    stroke_durations = []
    start = None
    stroke_count = 0

    for i in range(len(pressure)):
        if is_pen_down[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                duration = t[i-1] - t[start]
                stroke_durations.append(int(duration))
                start = None

    # Handle case where the last stroke goes to the end
    if start is not None:
        duration = t[-1] - t[start]
        stroke_durations.append(int(duration))
    stroke_durations = np.array(stroke_durations)
    # Calculate average stroke duration
    avg_stroke_duration = np.average(stroke_durations)
    
    # Calculate number of strokes
    stroke_count = len(stroke_durations)

    # Sign centroid
    pen_down = pressure > 0
    x_down = x[pen_down]
    y_down = y[pen_down]
    centroid_x = np.mean(x_down)
    centroid_y = np.mean(y_down)
    sign_centroid = np.array([centroid_x, centroid_y])

    sign_centroid, [pen_lifts], [stroke_count], [avg_stroke_duration], stroke_durations
    # convert to array of shape (num_frames, num_features)
    summary_features = np.zeros((1, 7))
    summary_features[0, :2] = sign_centroid
    summary_features[0, 4:7] = [pen_lifts, stroke_count, avg_stroke_duration]
    sign_feature_data = np.stack([x, y, pressure, azimuth, altitude, v, a], axis = 1)
    sign_feature_data = np.vstack([summary_features, sign_feature_data])

    # Convert to sliding window as a tensor for input to transformer model
    # cls token - added to let the transformer know it's a classification task. will add it to every sliding window.
    cls_token = sign_feature_data[0]
    feature_data_for_sign = sign_feature_data[1:]
    full_len = feature_data_for_sign.shape[0]
    stride = int(seq_len * (1 - overlap))

    sign_vector_with_windows = []

    for start in range(0, full_len, stride):
        end = start + seq_len - 1
        if start >= full_len:
            break
        sliding_win = feature_data_for_sign[start:end]

        # sliding win size <256 -1; -1 because we need to add the cls token as well
        if sliding_win.shape[0] < seq_len - 1:
            padding_len = seq_len - 1 - sliding_win.shape[0]
            padding = np.zeros((padding_len, feature_data_for_sign.shape[1]))
            sliding_win = np.vstack([sliding_win, padding])

        # sliding_win_tensor = torch.tensor(sliding_win, dtype = torch.float32)
        # cls_tensor = torch.tensor(cls_token, dtype = torch.float32)
        sliding_win = np.vstack([cls_token, sliding_win])

        # Create attention mask, to filter out padding when feeding to transformer
        data_len = min(seq_len - 1, full_len - start)
        attention_mask = torch.tensor([1] + [1] * data_len + [0] * (seq_len - 1 - full_len))
        sliding_win = torch.tensor(sliding_win)
        sign_vector_with_windows.append([sliding_win, attention_mask])
        if end >= full_len:
            break
    
    return sign_vector_with_windows

def normalize_for_eeg_related_data(data):
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    std[std == 0] = 1
    norm = (data - mean) / std
    return norm

def get_nth_difference_mean_for_signal(input_signal, n):
    diff = np.abs(input_signal[n:] - input_signal[:-n])
    res = np.sum(diff) / (input_signal.shape[0] - n)
    return res

def compute_freq_weighted_power_per_channel(channel, samp_freq, band):
    freqs, psd = welch(channel, fs=samp_freq, nperseg=len(channel))
    idx = (freqs >= band[0]) & (freqs <= band[1])
    freqs = freqs[idx]
    psd = psd[idx]
    return np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0

def get_freq_weighted_feature(signal, samp_freq, seq_len = 64, window = 2, overlap = 0.5, normalize = False):

    # over different frequency bands, calculate power
    # standard bands used for EEG - gamma (20-50 Hz), beta (13-20 Hz), alpha (8-13 Hz), theta (4-8 Hz), delta (0.5-4 Hz)
    # also takign windows of 2seconds witgh 1 second overlap
    # windows made using Hann window
    freq_bands = {
        'delta': [0.5, 4],
        'theta': [4, 8],
        'alpha': [8, 13],
        'beta': [13, 20],
        'gamma': [20, 50]
    }
    
    # Extract raw signal statistical features
    signal_mean = np.mean(np.array([np.mean(x) for x in signal]))
    signal_std = np.std(np.array(signal))
    
    
    first_difference_sample_mean_absolute_difference_raw_signal = get_nth_difference_mean_for_signal(signal, 1)
    second_difference_sample_mean_absolute_difference_raw_signal = get_nth_difference_mean_for_signal(signal, 2)
    
    normalized_signal = normalize_for_eeg_related_data(signal)
    first_difference_sample_mean_absolute_difference_normalized_signal = get_nth_difference_mean_for_signal(normalized_signal, 1)
    second_difference_sample_mean_absolute_difference_normalized_signal = get_nth_difference_mean_for_signal(normalized_signal, 2)


    n_channels, n_samples = signal.shape
    window_len = int(samp_freq * window)
    step = int(samp_freq * overlap)
    n_windows = (n_samples - window_len) // step + 1

    features = []
    for w in range(n_windows):
        start = w * step
        end = start + window_len
        window_features = []
        for channel in range(n_channels):
            segment = signal[channel, start:end]
            freqs, psd = welch(segment, fs = samp_freq, nperseg = window_len)
            for band_range in freq_bands.values():
                idx = (freqs >= band_range[0]) & (freqs <= band_range[1])
                bp = trapezoid(psd[idx], freqs[idx])
                window_features.append(bp)
            fwp = compute_freq_weighted_power_per_channel(segment, samp_freq=samp_freq, band = (0.5, 50))
            window_features.append(fwp)
        features.append(window_features)
    features = np.array(features)
    # print("Features shape before normalizing: ", features.shape)
    if normalize:
        features = normalize_for_eeg_related_data(features)
    # print(features.shape)

    cls_token = np.zeros(features.shape[1])
    cls_token[:6] = [signal_mean, signal_std, first_difference_sample_mean_absolute_difference_raw_signal, second_difference_sample_mean_absolute_difference_raw_signal, first_difference_sample_mean_absolute_difference_normalized_signal, second_difference_sample_mean_absolute_difference_normalized_signal]
    features = np.vstack([cls_token, features])
    return features

def modify_eeg_feature_data_for_transformer(eeg_feature_data):

    eeg_max_window = max([item[0].shape[0] for item in eeg_feature_data.values()])
    eeg_feature_vector_with_padding_and_attention = {}
    # print("Maximum n_windows found for EEG: ", eeg_max_window)
    for key, value in eeg_feature_data.items():
        # print("UserID: ", key)
        # print("Feature size: ", value[0].shape)
        current_user_eeg_feature_data = []
        for feature in value:
            # print("Feature: ", feature.shape)
            eeg_padded = feature
            if feature.shape[0] < eeg_max_window:
                padding = np.zeros((eeg_max_window - feature.shape[0], feature.shape[1]))
                # eeg_padded = torch.cat([feature, padding])
                eeg_padded = np.vstack([feature, padding])
            # The attention mask should have 1s for real tokens (including the cls_token) and 0s for padding
            eeg_padded = torch.tensor(eeg_padded)
            real_len = feature.shape[0]
            mask = np.array([1] * real_len + [0] * (eeg_max_window - real_len))
            attention_mask = torch.tensor(mask, dtype = torch.int64)
            current_user_eeg_feature_data.append([eeg_padded, attention_mask])
            # attention_mask = mask
        eeg_feature_vector_with_padding_and_attention[key] = current_user_eeg_feature_data
    return eeg_feature_vector_with_padding_and_attention

def get_eeg_features(mat_file):

    # mat_files_sorted = get_user_mat_data(user_id)
    mat_content = sp.loadmat(mat_file)
    
    # For debugging issues
    # to_print = mat_content['subject']
    # print(to_print)

    # converting ICA_EEG data into np structured array
    # eeg_columns = [i for i in mat_content['subject']['EEGHeader'][0][0][0].split(", ")]
    eeg_data_list = [i.tolist() for i in mat_content['subject']['ICA_EEG'][0][0]]

    # print(eeg_data_list)
    # eeg_input = pd.DataFrame(eeg_data_list).T
    # eeg_input.columns = eeg_columns

    norm_eeg_data = normalize_eeg_data(eeg_data_list)
    
    eeg_features_vector = get_freq_weighted_feature(norm_eeg_data, constants.eeg_samp_freq, normalize = True)
    return eeg_features_vector

def get_feature_vectors_for_all_users():
    csv_files, mat_files, user_ids = get_dataset_files_and_user_ids()
    sign_features_for_all_users = {}
    eeg_features_for_all_users = {}
    for user in user_ids:
        sign_features_for_all_users[user] = []
        eeg_features_for_all_users[user] = []
        user_csv_raw = [file for file in csv_files if user in file]
        user_mat_raw = [file for file in mat_files if user in file]
        # for debugging only
        print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("User ID: ", user)

        # Uncomment for debugging purposes only
        # print("CSV Data: ")
        # pp.pprint(user_csv_raw)
        # print("\n")
        # print("MAT Data:")
        # pp.pprint(user_mat_raw)

        for csv_file in user_csv_raw:
            sign_feature_vector = get_signature_feature_vector(csv_file, user_ids)
            # sign_feature_vector = modify_sign_data_for_transformer(sign_feature_vector)
            print("Extracting sign features for file: ", csv_file)
            # print("Sign feature vector: ")
            # pp.pprint(sign_feature_vector)
            sign_features_for_all_users[user].append(sign_feature_vector)

        for mat_file in user_mat_raw:
            eeg_feature_vector = get_eeg_features(mat_file)
            # eeg_feature_vector = pad_eeg_signal_for_transformer(eeg_feature_vector)
            print("Extracting EEG features for file: ", mat_file)
            eeg_features_for_all_users[user].append(eeg_feature_vector)
    eeg_features_for_all_users = modify_eeg_feature_data_for_transformer(eeg_features_for_all_users)
    return sign_features_for_all_users, eeg_features_for_all_users

def get_list_of_user_ids():
    user_ids = []
    for root, dir, files in os.walk(dataset_path):
        if os.path.basename(root) != 'Genuine' and os.path.basename(root) != 'Forged' and os.path.basename(root) != 'SignEEGv1.0':
            user_ids.append(os.path.basename(root))
    # print(len(user_ids))
    return user_ids
# sign_features_final, eeg_features_final = get_feature_vectors_for_all_users()
