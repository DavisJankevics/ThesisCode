import tensorflow as tf
import pandas as pd
import os
import librosa
import numpy as np

def load_audio_and_labels(audio_file_path, label_file_path, sr=44100, hop_length=512, n_fft=2048, n_mels=229, target_duration=10):
    target_length = int(sr * target_duration / hop_length)

    audio, _ = librosa.load(audio_file_path, sr=sr, mono=True)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
    start_index = 0

    if mel_spec_norm.shape[1] > target_length:
        max_start_index = mel_spec_norm.shape[1] - target_length
        start_index = np.random.randint(0, max_start_index)
        mel_spec_norm = mel_spec_norm[:, start_index:start_index + target_length]
    else:
        padding = -np.ones((n_mels, target_length - mel_spec_norm.shape[1]))
        mel_spec_norm = np.concatenate((mel_spec_norm, padding), axis=1)

    labels_df = pd.read_csv(label_file_path)
    label_tensor = np.zeros((target_length, 88), dtype=np.float32)

    for _, row in labels_df.iterrows():
        start_time = row['start_time'] / sr
        end_time = row['end_time'] / sr
        start_step = max(int(start_time * sr / hop_length) - start_index, 0)
        end_step = min(int(end_time * sr / hop_length) - start_index, target_length)
        note = int(row['note']) - 21
        
        if 0 <= note < 88 and start_step < end_step:
            label_tensor[start_step:end_step, note] = 1

    return mel_spec_norm.T, label_tensor

def create_tf_dataset(root_dir, split, sr=44100, hop_length=512, n_fft=2048, n_mels=229, target_duration=10):
    data_dir = os.path.join(root_dir, f'{split}_data')
    labels_dir = os.path.join(root_dir, f'{split}_labels')
    audio_files = sorted(os.listdir(data_dir))
    
    def gen():
        for audio_file in audio_files:
            audio_path = os.path.join(data_dir, audio_file)
            label_path = os.path.join(labels_dir, audio_file.replace('.wav', '.csv'))
            features, labels = load_audio_and_labels(audio_path, label_path, sr, hop_length, n_fft, n_mels, target_duration)
            yield (features, labels)

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, n_mels), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 88), dtype=tf.float32),
        ))
