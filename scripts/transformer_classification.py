import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import constants
import os
from dotenv import load_dotenv
import matplotlib.cm as cm
import scipy.io as sp
import pprint as pp
from scipy.signal import welch
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

# some calculations
    # for the dataset, 10 seconds = 1280 frames. 1 second = 128 frames. We can sue this to find the number of seconds that were taken to record the signature.

    recording_samp_rate = 128 # per second
    per_phase_frames = 1280 # seconds
    max_seq_len_for_data = 3000 # frames
    # desampling_factor = 1 # reducing sequences by this size

def get_dataset_files_and_user_ids(data_category, data_type = constants.TRAIN):
    user_ids = []
    labels = []
    files_mat = []
    
    # Get training and testing data
    data_split = pd.read_csv(os.path.join(dataset_path, "Identification_split.csv"))
    files_for_task = list(data_split[data_split.set == data_type].filename)
    data_categories = [constants.GENUINE, constants.FORGED] if data_category == constants.ALL else [data_category]

    for root, dirs, files in os.walk(dataset_path):
        if os.path.basename(root) in data_categories:
            for file in files:
                if file.endswith('.mat') and file in files_for_task:
                    files_mat.append(os.path.join(root, file))
                    labels.append(os.path.basename(root))
        if os.path.basename(root) != constants.GENUINE and os.path.basename(root) != constants.FORGED and os.path.basename(root) != 'SignEEGv1.0':
            user_ids.append(os.path.basename(root))
        
    return files_mat, user_ids, labels

def get_sig_eeg_raw_data(mat_files, labels, desampling_factor = 1):
    raw_data_list = []
    for mat_file, label in zip(mat_files, labels):
        mat_content = sp.loadmat(mat_file)
        user_id = str(mat_content['subject']['SubjectID'][0][0][0])
        sig_data = mat_content['subject']['SignWacom'][0][0]
        eeg_ica_data = mat_content['subject']['ICA_EEG'][0][0].T
        sig_data = torch.from_numpy(np.delete(sig_data, 0, axis=1)).to(dtype=torch.float32)
        
        # getting part of eeg data during which signature was recorded (ROI)
        roi_frames_start = -(eeg_ica_data.shape[0] % per_phase_frames) if per_phase_frames > 0 else 0
        eeg_ica_data = torch.from_numpy(eeg_ica_data[roi_frames_start:]).to(dtype=torch.float32)

        # desampling the data
        if desampling_factor > 1:
            sig_data = sig_data[::desampling_factor, :]
            eeg_ica_data = eeg_ica_data[::desampling_factor, :]

        if sig_data.shape[0] > max_seq_len_for_data:
            # print("Caught you!!!")
            # print("User ID: ", user_id)
            # print("File: ", mat_file)  
            continue # Skip these files because it's too long, outlier
        raw_data_list.append({
            'sign_data': sig_data,
            'eeg_data': eeg_ica_data,
            'user_id': user_id,
            'label': 0 if label == constants.GENUINE else 1,
            'file': mat_file
        })

    return raw_data_list

def augment_sign_data(sign_data, noise_std=0.01, scale_range=(0.95, 1.05), rotation_deg=5):
    augmented = sign_data.clone()
    augmented[:, 2:] += torch.randn_like(augmented[:, 2:]) * noise_std
    scale = random.uniform(*scale_range)
    augmented[:, 2] *= scale
    augmented[:, 3] *= scale

    # Random rotation (x, y)
    theta = math.radians(random.uniform(-rotation_deg, rotation_deg))
    x = augmented[:, 2].clone()
    y = augmented[:, 3].clone()
    augmented[:, 2] = x * math.cos(theta) - y * math.sin(theta)
    augmented[:, 3] = x * math.sin(theta) + y * math.cos(theta)

    return augmented

def augment_eeg_data(eeg_data, noise_std=0.01, scale_range=(0.95, 1.05), time_shift_max=10):
    augmented = eeg_data.clone()
    augmented += torch.randn_like(augmented) * noise_std
    scale = random.uniform(*scale_range)
    augmented *= scale
    shift = random.randint(-time_shift_max, time_shift_max)
    if shift > 0:
        augmented = torch.cat([augmented[shift:], torch.zeros_like(augmented[:shift])], dim=0)
    elif shift < 0:
        augmented = torch.cat([torch.zeros_like(augmented[shift:]), augmented[:shift]], dim=0)

    return augmented

def normalize_sign_data_dict(sign_data):

    mean = torch.mean(sign_data[:, 2:], dim=0)
    std = torch.std(sign_data[:, 2:], dim=0)
    std = torch.where(std == 0, torch.tensor(1.0, dtype=torch.float32), std)
    normalized = (sign_data[:, 2:] - mean) / std
    normalized = torch.cat([sign_data[:, 0:2], normalized], dim=1).to(dtype=torch.float32)
    return normalized

def get_sign_data_features(sign_data):
    normalized_sign_data = normalize_sign_data_dict(sign_data)
    x = sign_data[:, 2]
    y = sign_data[:, 3]

    normalized_sign_data = torch.tensor(normalized_sign_data, dtype=torch.float32)
    norm_x = normalized_sign_data[:, 2]
    norm_y = normalized_sign_data[:, 3]
    vx = torch.gradient(norm_x)[0]
    vy = torch.gradient(norm_y)[0]
    velocity = torch.sqrt(vx**2 + vy**2)
    ax = torch.gradient(vx)[0]
    ay = torch.gradient(vy)[0]
    acceleration = torch.sqrt(ax**2 + ay**2)
    
    avg_vx = torch.mean(vx)
    avg_vy = torch.mean(vy)
    avg_ax = torch.mean(ax)
    avg_ay = torch.mean(ay)
    
    # log curvature radius
    dt = 1
    dx = torch.gradient(norm_x, spacing=(dt,))[0]
    dy = torch.gradient(norm_y, spacing=(dt,))[0]
    v_t = torch.sqrt(dx ** 2 + dy ** 2)
    v_t = torch.where(v_t == 0, torch.tensor(1e-10, dtype=v_t.dtype), v_t)
    theta = torch.atan2(dy, dx)
    dtheta = torch.gradient(theta, spacing=(dt,))[0]
    dtheta = torch.where(dtheta == 0, torch.tensor(1e-10, dtype=dtheta.dtype), dtheta)
    log_curv_radius = torch.log(torch.abs(v_t / dtheta) + 1e-10)
    # print("Log Curve Radius shape: ", log_curv_radius.shape)
    # getting static features
    pendown_frames = normalized_sign_data[:, 1] == 1
    num_strokes = torch.unique(normalized_sign_data[pendown_frames][:, 0]).shape[0]
    x_down = normalized_sign_data[pendown_frames][:, 2]
    y_down = normalized_sign_data[pendown_frames][:, 3]
    sign_centroid = torch.tensor([torch.mean(x_down), torch.mean(y_down)], dtype=torch.float32)
    if y_down.shape[0] > 0:
        sign_height = torch.max(y_down) - torch.min(y_down)
    else:
        sign_height = 0
    if x_down.shape[0] > 0:
        sign_width = torch.max(x_down) - torch.min(x_down)
    else:
        sign_width = 0
    height_width_ratio = sign_height / sign_width if sign_width != 0 else torch.tensor(0.0, dtype=torch.float32)
    
    # new time dependent feature - jerk
    jerk = torch.sqrt(torch.gradient(vx)[0]**2 + torch.gradient(vy)[0]**2)

    pressure = sign_data[pendown_frames][:, 4]
    azimuth = sign_data[pendown_frames][:, 5]
    altitude = sign_data[pendown_frames][:, 6]
    avg_pressure = torch.mean(pressure)
    avg_azimuth = torch.mean(azimuth)
    avg_altitude = torch.mean(altitude)
    max_pressure = torch.max(pressure) if pressure.numel() > 0 else torch.tensor(0.0, dtype=torch.float32)
    sign_duration = sign_data.shape[0] / recording_samp_rate
    cls_token = torch.tensor([
        num_strokes, sign_height, sign_width, height_width_ratio, sign_centroid[0], sign_centroid[1], avg_pressure, avg_azimuth, avg_altitude, avg_vx, avg_vy, avg_ax, avg_ay, max_pressure, sign_duration], dtype=torch.float32)
    sign_data_aug = torch.cat([normalized_sign_data, velocity.unsqueeze(1), acceleration.unsqueeze(1), log_curv_radius.unsqueeze(1), jerk.unsqueeze(1)], dim=1)

    return sign_data_aug, cls_token

def attach_attention_tokens_and_padding(data, max_len):
    # print("EEG Data shape: ", data.shape)
    if data.shape[0] == 0:
        feat_dim = data.shape[1] if data.ndim == 2 else 1
        return torch.zeros((max_len, feat_dim), dtype=torch.float32), torch.zeros(max_len, dtype=torch.float32)
    seq_len, feat_dim = data.shape
    pad_width = (0, max_len - seq_len)
    padded_data = torch.nn.functional.pad(data, (0, 0, 0, pad_width[1]), mode='constant', value=0)
    attention_mask = torch.zeros(max_len + 1, dtype=torch.float32)
    attention_mask[:seq_len + 1] = 1  # +1 for cls_token
    return padded_data, attention_mask

def normalize_eeg_data_dict(eeg_data_dict):
    normalized_eeg_data_dict = {}
    for user_id, eeg_list in eeg_data_dict.items():
        normalized_eeg_data_dict[user_id] = []
        for eeg_data in eeg_list:
            mean = eeg_data.mean(dim=0, keepdim=True)
            std = eeg_data.std(dim=0, keepdim=True)
            std = torch.where(std == 0, torch.tensor(1.0, dtype=std.dtype, device=std.device), std)
            normalized = (eeg_data - mean) / std
            normalized_eeg_data_dict[user_id].append(normalized)
    return normalized_eeg_data_dict

def extract_fft_features(eeg_data, fs=128, epoch_length_sec=1): # taking 1s instead of 30s because the signal duration is short
    n_samples, n_channels = eeg_data.shape
    epoch_len = int(epoch_length_sec * fs)
    n_epochs = n_samples // epoch_len
    features = []
    for i in range(n_epochs):
        epoch = eeg_data[i*epoch_len:(i+1)*epoch_len, :]
        epoch_features = []
        for ch in range(n_channels):
            fft_vals = np.fft.rfft(epoch[:, ch])
            fft_power = np.abs(fft_vals)
            epoch_features.extend(np.abs(fft_vals).flatten())
        features.append(epoch_features)
    return features

def get_nth_difference_mean_for_signal(input_signal, n):
    input_signal = torch.as_tensor(input_signal)
    diff = torch.abs(input_signal[n:] - input_signal[:-n])
    res = torch.sum(diff) / (input_signal.shape[0] - n)
    return res

def normalize_for_eeg_related_data(data):
    data = torch.as_tensor(data, dtype=torch.float32)
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)
    std = torch.where(std == 0, torch.tensor(1.0, dtype=std.dtype, device=std.device), std)
    norm = (data - mean) / std
    return norm

def get_eeg_data_features(eeg_data, fs=recording_samp_rate):

    signal_mean = torch.mean(eeg_data)
    signal_std = torch.std(eeg_data)

    first_difference_sample_mean_absolute_difference_raw_signal = get_nth_difference_mean_for_signal(eeg_data, 1)
    second_difference_sample_mean_absolute_difference_raw_signal = get_nth_difference_mean_for_signal(eeg_data, 2)

    normalized_signal = normalize_for_eeg_related_data(eeg_data)
    first_difference_sample_mean_absolute_difference_normalized_signal = get_nth_difference_mean_for_signal(normalized_signal, 1)
    second_difference_sample_mean_absolute_difference_normalized_signal = get_nth_difference_mean_for_signal(normalized_signal, 2)
    fw_powers = []
    eeg_data = torch.as_tensor(eeg_data, dtype=torch.float32)
    for ch in range(eeg_data.shape[1]):
        # Welch returns numpy arrays, so convert to torch
        f, Pxx = welch(eeg_data[:, ch].cpu().numpy(), fs=fs)
        f = torch.from_numpy(f).to(eeg_data.device)
        Pxx = torch.from_numpy(Pxx).to(eeg_data.device)
        fw_power = torch.sum(f * Pxx) / torch.sum(Pxx) if torch.sum(Pxx) > 0 else torch.tensor(0.0, device=eeg_data.device)
        fw_powers.append(fw_power)
    fw_power_arr = torch.stack(fw_powers).unsqueeze(0)
    # cls_token = torch.cat([signal_mean, signal_std, first_difference_sample_mean_absolute_difference_raw_signal, second_difference_sample_mean_absolute_difference_raw_signal, first_difference_sample_mean_absolute_difference_normalized_signal, second_difference_sample_mean_absolute_difference_normalized_signal])
    # cls_token = torch.stack(fw_power_arr)
    fft_features = extract_fft_features(eeg_data.cpu().numpy(), fs=fs) # because we are using np.fft.rfft
    features = [signal_mean, signal_std, first_difference_sample_mean_absolute_difference_raw_signal, second_difference_sample_mean_absolute_difference_raw_signal, first_difference_sample_mean_absolute_difference_normalized_signal, second_difference_sample_mean_absolute_difference_normalized_signal]
    # for epoch_feat in fft_features:
    #     features.extend(epoch_feat)
    features.extend(fw_power_arr.squeeze(0).tolist())
    cls_token = torch.tensor(features, dtype=torch.float32)
    # uncomment if fft_features as eeg_data fails
    # return normalized_signal, cls_token
    fft_features = torch.tensor(fft_features, dtype=torch.float32)
    fft_features = torch.nan_to_num(fft_features, nan=0.0, posinf=0.0, neginf=0.0)
    return fft_features, cls_token

class SignatureEEGDataset(Dataset):
    def __init__(self, input_data, num_classes):
        sign_data = input_data['sign_data']
        eeg_data = input_data['eeg_data']
        sign_attention_masks = input_data['sign_attention_masks']
        eeg_attention_masks = input_data['eeg_attention_masks']
        sign_cls_tokens = input_data['sign_cls_tokens']
        eeg_cls_tokens = input_data['eeg_cls_tokens']
        labels = input_data['labels']

        self.sign_x_ts = sign_data
        self.sign_cls_token = sign_cls_tokens
        self.sign_attention_mask = sign_attention_masks
        self.sign_seq_len = sign_data[0].shape[0]
        self.sign_ts_dim = sign_data[0].shape[1]

        self.eeg_x_ts = eeg_data
        self.eeg_cls_token = eeg_cls_tokens
        self.eeg_attention_mask = eeg_attention_masks
        self.eeg_seq_len = eeg_data[0].shape[0]
        self.eeg_ts_dim = eeg_data[0].shape[1]

        self.num_classes = num_classes
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'sign_x_ts': self.sign_x_ts[idx],
            'sign_cls_token': self.sign_cls_token[idx],
            'sign_attention_mask': self.sign_attention_mask[idx],
            'eeg_x_ts': self.eeg_x_ts[idx],
            'eeg_cls_token': self.eeg_cls_token[idx],
            'eeg_attention_mask': self.eeg_attention_mask[idx],
            'labels': self.labels[idx],
        }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout = 0.1):
        super().__init__()
        pe = torch.zeros(max_len + 1, d_model)
        position = torch.arange(0, max_len + 1, dtype = torch.float).unsqueeze(1)
        divterm = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # Credits to hkproj@github for this as https://github.com/hkproj/pytorch-transformer/blob/main/model.py
        pe[:, 0::2] = torch.sin(position * divterm)
        pe[:, 1::2] = torch.cos(position * divterm)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.pe[:, :x.shape[1]]
    
class SignatureEEGTransformer(nn.Module):
    def __init__(self, sign_input_dim, sign_cls_dim, eeg_input_dim, eeg_cls_dim, d_model, num_classes, num_heads, num_layers, sign_max_seq_len, eeg_max_seq_len, dropout = 0.1):
        super().__init__()
        self.sign_transfomer = SignatureTransformer(sign_input_dim, sign_cls_dim, d_model, num_classes, num_heads, num_layers, sign_max_seq_len, dropout)
        self.eeg_transformer = SignatureTransformer(eeg_input_dim, eeg_cls_dim, d_model, num_classes, num_heads, num_layers, eeg_max_seq_len, dropout)

        self.classifier = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, num_classes))

    def forward(self, sign_x_ts, sign_cls_token, eeg_x_ts, eeg_cls_token, sign_attn_mask = None, eeg_attn_mask = None):
        sign_cls = self.sign_transfomer(sign_x_ts, sign_cls_token, sign_attn_mask)
        eeg_cls = self.eeg_transformer(eeg_x_ts, eeg_cls_token, eeg_attn_mask)
        multimodal_cls_output = torch.cat([sign_cls, eeg_cls], dim = 1)

        logits = self.classifier(multimodal_cls_output)
        return logits
    
class SignatureTransformer(nn.Module):
    def __init__(self, input_dim, cls_dim, d_model, num_classes, num_heads, num_layers, max_seq_len, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.cls_proj = nn.Linear(cls_dim, d_model)
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout = dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)
        # uncomment for single modality
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_classes))

    def forward(self, x_ts, cls_token, attn_mask = None):
        x_ts = torch.nan_to_num(x_ts, nan=0.0, posinf=0.0, neginf=0.0)
        cls_token = torch.nan_to_num(cls_token, nan=0.0, posinf=0.0, neginf=0.0)
        batch_size, t, feat_dim = x_ts.shape
        x_proj = self.input_projection(x_ts)
        cls_proj = self.cls_proj(cls_token).unsqueeze(1)
        # print("x_proj size: ", x_proj.shape)
        # print("cls_proj size: ", cls_proj.shape)
        x = torch.cat([cls_proj, x_proj], dim=1)
        # print("x_proj and cls_proj concatenated size: ", x.shape)
        x = x + self.positional_encoding(x)

        if attn_mask is not None:
            attn_mask = attn_mask == 0 # True = ignore the value, False = include it!!!!!!!!!!
            # cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=attn_mask.device)
            full_mask = torch.cat([attn_mask], dim=1)  # [batch_size, t+1]
        else:
            full_mask = None
        # print("Mask shape: ", full_mask.shape)
        x = self.transformer(x, src_key_padding_mask=full_mask)
        cls_output = x[:, 0, :]
        # uncomment for single modality transformer
        # logits = self.classifier(cls_output)
        # return logits

        # uncomment for multimodal transformer
        return cls_output 

if __name__ == "__main__":

    load_dotenv()
    dataset_path = os.getenv('DATASET_PATH')

    files_mat_genuine, user_ids_genuine, genuine_labels = get_dataset_files_and_user_ids(data_category=constants.GENUINE)
    files_mat_forged, user_ids_forged, forged_labels = get_dataset_files_and_user_ids(data_category=constants.FORGED)

    files_mat_genuine.extend(files_mat_forged)
    files_mat_appended = files_mat_genuine
    genuine_labels.extend(forged_labels)
    labels_appended = genuine_labels

    # shuffling to prevent overfitting
    files_all = np.array(files_mat_appended)
    labels_all = np.array(labels_appended)

    indices = np.arange(len(files_all))
    np.random.shuffle(indices)

    files_mat_appended = files_all[indices]
    labels_appended = labels_all[indices]

    raw_data = get_sig_eeg_raw_data(files_mat_appended, labels_appended)

    # for debugging only
    print("EEG Data seq len: ", [data['eeg_data'].shape[0] for data in raw_data])

    augmented_raw_data = []
    num_augments = 3

    for sample in raw_data:
        augmented_raw_data.append(sample)
        for _ in range(num_augments):
            aug_sample = sample.copy()
            aug_sample['sign_data'] = augment_sign_data(sample['sign_data'])
            aug_sample['eeg_data'] = augment_eeg_data(sample['eeg_data'])
            augmented_raw_data.append(aug_sample)

    raw_data = augmented_raw_data

    for i in range(len(raw_data)):
        sign_data_with_features, sign_cls_token = get_sign_data_features(raw_data[i]['sign_data'])
        eeg_data_with_features, eeg_cls_token = get_eeg_data_features(raw_data[i]['eeg_data'])
        # print("EEG Feature data shape: ", eeg_data_with_features.shape)
        raw_data[i]['sign_data'] = sign_data_with_features
        raw_data[i]['sign_cls_token'] = sign_cls_token
        raw_data[i]['eeg_data'] = eeg_data_with_features
        raw_data[i]['eeg_cls_token'] = eeg_cls_token

    # sign_max_seq_len = max([data['sign_data'].shape[0] for data in raw_data])
    # eeg_max_seq_len = max([data['eeg_data'].shape[0] for data in raw_data])

    sign_max_seq_len = max_seq_len_for_data
    eeg_max_seq_len = 10

    for i in range(len(raw_data)):
        sign_data = raw_data[i]['sign_data']
        eeg_data = raw_data[i]['eeg_data']
        sign_data, sign_attention_mask = attach_attention_tokens_and_padding(sign_data, sign_max_seq_len)
        eeg_data, eeg_attention_mask = attach_attention_tokens_and_padding(eeg_data, eeg_max_seq_len)
        raw_data[i]['sign_data'] = sign_data
        raw_data[i]['eeg_data'] = eeg_data
        raw_data[i]['sign_attention_mask'] = sign_attention_mask
        raw_data[i]['eeg_attention_mask'] = eeg_attention_mask

    sign_data = [data['sign_data'] for data in raw_data]
    eeg_data = [data['eeg_data'] for data in raw_data]
    sign_attention_masks = [data['sign_attention_mask'] for data in raw_data]
    eeg_attention_masks = [data['eeg_attention_mask'] for data in raw_data]
    sign_cls_tokens = [data['sign_cls_token'] for data in raw_data]
    eeg_cls_tokens = [data['eeg_cls_token'] for data in raw_data]
    labels = [data['label'] for data in raw_data]
    files = [data['file'] for data in raw_data]

    input_data = {
        'sign_data': sign_data,
        'eeg_data': eeg_data,
        'sign_attention_masks': sign_attention_masks,
        'eeg_attention_masks': eeg_attention_masks,
        'sign_cls_tokens': sign_cls_tokens,
        'eeg_cls_tokens': eeg_cls_tokens,
        'labels': labels,
    }
    num_classes = 2
    multimodal_dataset = SignatureEEGDataset(input_data, num_classes)
    batch_size = 16
    multimodal_dataloader = DataLoader(multimodal_dataset, batch_size=batch_size, shuffle=True)
    sign_ts_dim = input_data['sign_data'][0].size(1)
    sign_cls_dim = input_data['sign_cls_tokens'][0].size(0)
    sign_seq_len = input_data['sign_data'][0].size(0)
    eeg_ts_dim = input_data['eeg_data'][0].size(1)
    eeg_cls_dim = input_data['eeg_cls_tokens'][0].size(0)
    eeg_seq_len = input_data['eeg_data'][0].size(0)

    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) and labels.is_cuda else (
        labels.numpy() if isinstance(labels, torch.Tensor) else labels
    )

    train_indices, val_indices = train_test_split(
        range(len(labels)), test_size=0.2, random_state=42, stratify=labels
    )
    train_data = [raw_data[i] for i in train_indices]
    val_data = [raw_data[i] for i in val_indices]

    def build_input(data):
        return {
            'sign_data': [d['sign_data'] for d in data],
            'eeg_data': [d['eeg_data'] for d in data],
            'sign_attention_masks': [d['sign_attention_mask'] for d in data],
            'eeg_attention_masks': [d['eeg_attention_mask'] for d in data],
            'sign_cls_tokens': [d['sign_cls_token'] for d in data],
            'eeg_cls_tokens': [d['eeg_cls_token'] for d in data],
            'labels': [d['label'] for d in data],
        }

    train_input = build_input(train_data)
    val_input = build_input(val_data)

    train_dataset = SignatureEEGDataset(train_input, num_classes=2)
    val_dataset = SignatureEEGDataset(val_input, num_classes=2)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SignatureEEGTransformer(
        sign_input_dim=sign_ts_dim, sign_cls_dim=sign_cls_dim,
        eeg_input_dim=eeg_ts_dim, eeg_cls_dim=eeg_cls_dim,
        d_model=128, num_classes=2, num_heads=4, num_layers=2,
        sign_max_seq_len=sign_seq_len, eeg_max_seq_len=eeg_seq_len
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_labels, train_preds = 0, [], []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
            sign_x_ts = batch['sign_x_ts'].to(device)
            sign_cls_token = batch['sign_cls_token'].to(device)
            sign_attention_mask = batch['sign_attention_mask'].to(device)
            eeg_x_ts = batch['eeg_x_ts'].to(device)
            eeg_cls_token = batch['eeg_cls_token'].to(device)
            eeg_attention_mask = batch['eeg_attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(sign_x_ts, sign_cls_token, eeg_x_ts, eeg_cls_token, sign_attention_mask, eeg_attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(preds.cpu().numpy())

        avg_train_loss = train_loss / len(train_labels)
        train_acc = accuracy_score(train_labels, train_preds)

        model.eval()
        val_loss, val_labels, val_preds = 0, [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False):
                sign_x_ts = batch['sign_x_ts'].to(device)
                sign_cls_token = batch['sign_cls_token'].to(device)
                sign_attention_mask = batch['sign_attention_mask'].to(device)
                eeg_x_ts = batch['eeg_x_ts'].to(device)
                eeg_cls_token = batch['eeg_cls_token'].to(device)
                eeg_attention_mask = batch['eeg_attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(sign_x_ts, sign_cls_token, eeg_x_ts, eeg_cls_token, sign_attention_mask, eeg_attention_mask)
                loss = loss_fn(logits, labels)
                val_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

        avg_val_loss = val_loss / len(val_labels)
        val_acc = accuracy_score(val_labels, val_preds)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print("="*50)
    torch.save(model.state_dict(), os.path.join(os.getenv("MODEL_PATH"), f"multimodal_model_{datetime.now().strftime('%m%d%Y-%H%M%S')}.pth"))