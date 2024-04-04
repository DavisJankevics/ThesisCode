import os
import numpy as np
import pretty_midi
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

def midi_to_label_tensor(midi_path, sr=44100, hop_length=int(44100 * (1/64))):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    max_end_time = max(note.end for instrument in midi_data.instruments for note in instrument.notes)
    target_length = int(max_end_time * sr / hop_length) + 1

    label_tensor = np.zeros((target_length, 88), dtype=np.float32)

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start_time = note.start
            end_time = note.end
            pitch = note.pitch - 21
            
            start_step = int(start_time * sr / hop_length)
            end_step = int(end_time * sr / hop_length)
            
            if 0 <= pitch < 88:
                start_step = max(start_step, 0)
                end_step = min(end_step, target_length)
                
                label_tensor[start_step:end_step, pitch] = 1

    return label_tensor

def calculate_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

def pad_sequences_to_match(gt_tensor, pred_tensor):
    max_length = max(gt_tensor.shape[0], pred_tensor.shape[0])
    gt_padded = np.zeros((max_length, gt_tensor.shape[1]))
    pred_padded = np.zeros((max_length, pred_tensor.shape[1]))
    
    gt_padded[:gt_tensor.shape[0], :gt_tensor.shape[1]] = gt_tensor
    pred_padded[:pred_tensor.shape[0], :pred_tensor.shape[1]] = pred_tensor
    
    return gt_padded, pred_padded

def evaluate_transcription_tensor_with_tf_metrics(gt_folder, test_folders):
    precision_metric = Precision()
    recall_metric = Recall()
    binary_accuracy_metric = BinaryAccuracy()

    for test_folder in test_folders:
        print(f"Evaluating test folder: {test_folder}")
        precision_metric.reset_states()
        recall_metric.reset_states()
        binary_accuracy_metric.reset_states()

        for gt_file in os.listdir(gt_folder):
            if gt_file.endswith(".mid"):
                gt_number = gt_file.split('_')[0]
                test_files = [f for f in os.listdir(test_folder) if f.startswith(gt_number) and f.endswith(".mid")]

                for test_file in test_files:
                    gt_path = os.path.join(gt_folder, gt_file)
                    test_path = os.path.join(test_folder, test_file)

                    gt_tensor = midi_to_label_tensor(gt_path)
                    pred_tensor = midi_to_label_tensor(test_path)

                    gt_padded, pred_padded = pad_sequences_to_match(gt_tensor, pred_tensor)

                    precision_metric.update_state(gt_padded, pred_padded)
                    recall_metric.update_state(gt_padded, pred_padded)
                    binary_accuracy_metric.update_state(gt_padded, pred_padded)

        precision = precision_metric.result().numpy()
        recall = recall_metric.result().numpy()
        f1_score = calculate_f1_score(precision, recall)

        print(f"\nSummary for {test_folder}:")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1_score:.3f}")
        print(f"Binary Accuracy: {binary_accuracy_metric.result().numpy():.3f}\n")

evaluate_transcription_tensor_with_tf_metrics('AMT\MyAMT\src\database\musicnet/test_midis', 
                       ['output/baseline\omnizart', 'output\mss\omnizart\combined',
                        'output/baseline\mt3', 'output\mss\mt3\combined',
                        'output/baseline\my', 'output\mss\my\combined']
                        )