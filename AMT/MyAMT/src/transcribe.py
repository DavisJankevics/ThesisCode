import argparse
import os
from utils.utils import preprocess_audio, prediction_to_midi
from model.model import build_model
from conf.conf import Config
import tensorflow as tf
import numpy as np
import glob

def load_and_configure_model(checkpoint_location, config):
    input_shape = (None, config.input_size)
    num_notes = config.output_size

    model = build_model(input_shape, num_notes, config)

    model = tf.keras.models.load_model(checkpoint_location)
    return model

def transcribe(audio_path, model, output_path, config):
    audio, tempo = preprocess_audio(audio_path, config)
    audio = np.expand_dims(audio, axis=0)

    prediction = model.predict(audio)
    prediction_to_midi(prediction, tempo, output_path)

def process_folder(input_folder, model, output_folder, config):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for audio_path in glob.glob(os.path.join(input_folder, '*.wav')):
        file_name = os.path.basename(audio_path)
        output_path = os.path.join(output_folder, file_name.replace('.wav', '.mid'))
        print(f"Transcribing {audio_path} to {output_path}")
        transcribe(audio_path, model, output_path, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcribe audio files in a folder using a trained model')
    parser.add_argument('--input_folder', '-i', type=str, required=True, help='Path to the folder containing audio files to transcribe')
    parser.add_argument('--checkpoint_path', '-c', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--output_folder', '-o', type=str, required=True, help='Path to the folder where output MIDI files will be saved')
    args = parser.parse_args()

    config = Config()
    model = load_and_configure_model(args.checkpoint_path, config)

    process_folder(args.input_folder, model, args.output_folder, config)
