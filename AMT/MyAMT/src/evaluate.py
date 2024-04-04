import argparse
import tensorflow as tf
from conf.conf import Config
from data.data import create_tf_dataset
from model.model import build_model
from tensorflow.keras.models import load_model

def evaluate(db_location, model_path):
    config = Config()
    input_shape = (None, config.input_size)
    num_notes = config.output_size

    model = build_model(input_shape, num_notes, config)

    print(f"Loading model from {model_path}.")
    model = load_model(model_path)
    print("Loading test dataset.")
    test_dataset = create_tf_dataset(root_dir=db_location, split='test')
    test_dataset = test_dataset.batch(config.batch_size)

    print("Evaluating on test dataset.")
    results = model.evaluate(test_dataset)

    metrics = ['loss', 'binary_accuracy', 'precision', 'recall']
    for metric, result in zip(metrics, results):
        print(f"{metric}: {result:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate BiLSTM for Music Transcription using TensorFlow')
    parser.add_argument('--db_location', type=str, required=True, help='Location of MusicNet database')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model for evaluation')

    args = parser.parse_args()

    evaluate(args.db_location, args.model_path)
