import librosa
import numpy as np
from conf.conf import Config
import pretty_midi

def preprocess_audio(audio_path, config):
    audio, _ = librosa.load(audio_path, sr=config.sr, mono=True)
    tempo, _ = librosa.beat.beat_track(y=audio, sr=config.sr)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=config.sr, n_fft=config.n_fft, hop_length=config.hop_length, n_mels=config.n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
    return mel_spec_norm.T, tempo

def prediction_to_midi(predictions, tempo, output_path="output.mid"):
    midi_data = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    predictions = np.squeeze(predictions)
    predictions_transformed = transform_predictions(predictions)

    for note_info in predictions_transformed:
        note, start_time, duration = note_info
        midi_note = pretty_midi.Note(
            velocity=100,
            pitch=note + 21,
            start=start_time,
            end=start_time + duration
        )
        piano.notes.append(midi_note)

    midi_data.instruments.append(piano)
    midi_data.write(output_path)

def transform_predictions(predictions, threshold=0.5, config = Config()):
    results = []
    active_notes = {note: None for note in range(88)}

    for time_step in range(predictions.shape[0]):
        for note in range(predictions.shape[1]):
            activation = predictions[time_step, note]

            if activation > threshold and active_notes[note] is None:
                active_notes[note] = time_step * config.hop_length / config.sr

            elif activation <= threshold and active_notes[note] is not None:
                start_time = active_notes[note]
                duration = (time_step * config.hop_length / config.sr) - start_time
                results.append((note, start_time, duration))
                active_notes[note] = None

    for note, start_time in active_notes.items():
        if start_time is not None:
            duration = (predictions.shape[0]* config.hop_length / config.sr) - start_time
            results.append((note, start_time, duration))

    return results