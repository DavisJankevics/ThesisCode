import argparse
from model.model import build_model
from conf.conf import Config
from data.data import create_tf_dataset
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryFocalCrossentropy

class BatchMetricsLogger(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        print(f"")
        
    def on_test_batch_end(self, batch, logs=None):
        logs = logs or {}
        print(f"\nValidation Batch {batch}, Loss: {logs.get('loss')}, Accuracy: {logs.get('binary_accuracy')}, Precision: {logs.get('precision')}, Recall: {logs.get('recall')}")

def train(db_location, load_model_path=None):
    config = Config()
    input_shape = (None, config.input_size)
    num_notes = config.output_size

    model = build_model(input_shape, num_notes, config)
    initial_epoch = 0
    
    if load_model_path:
        filename = load_model_path.split('/')[-1]
        epoch_str = filename.split('_')[-1]
        initial_epoch = int(epoch_str.split('.')[0])
        print(f"Loading model from {load_model_path} at epoch {initial_epoch}.")
        if load_model_path.endswith('.h5'):
            model = tf.keras.models.load_model(load_model_path)
            print(f"Model loaded successfully from {load_model_path}.")
        else:
            model.load_weights(load_model_path)
            print(f"Weights loaded successfully from {load_model_path}.")
    else:
        print("Starting training with a new model.")
        optimizer = Adam()
        loss_function = BinaryFocalCrossentropy(gamma=config.gamma,alpha=config.alpha, apply_class_balancing=True)
        accuracy = BinaryAccuracy(name = 'binary_accuracy', threshold = 0.5)
        model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy, Precision(thresholds = 0.5), Recall(thresholds = 0.5)])

    train_dataset = create_tf_dataset(root_dir=db_location, split='train', sr=config.sr, hop_length=config.hop_length, n_fft=config.n_fft, n_mels=config.n_mels, target_duration=config.target_duration)
    val_dataset = create_tf_dataset(root_dir=db_location, split='validation', sr=config.sr, hop_length=config.hop_length, n_fft=config.n_fft, n_mels=config.n_mels, target_duration=config.target_duration)
    callbacks = [
        ModelCheckpoint("/checkpoints_{epoch:03d}.h5", save_weights_only=False, save_best_only=False, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, min_delta=0, restore_best_weights=True, verbose=1, mode='auto'),
        BatchMetricsLogger()
    ]
    history = model.fit(
        train_dataset.batch(config.batch_size),
        epochs=config.num_epochs,
        validation_data=val_dataset.batch(config.batch_size),
        callbacks=[callbacks],
        initial_epoch=initial_epoch
    )
    model.save("./final_model.h5")
    model.save_weights("./final_model_w.ckpt")

    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BiLSTM for Music Transcription')
    parser.add_argument('--db_location', type=str, required=True, help='Location of MusicNet database')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to load the model checkpoint')
    args = parser.parse_args()

    train(args.db_location, args.load_model_path)
