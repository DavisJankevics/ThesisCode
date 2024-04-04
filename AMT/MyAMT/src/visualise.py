import pretty_midi
import matplotlib.pyplot as plt

def visualize_midi(file_path):
    midi_data = pretty_midi.PrettyMIDI(file_path)

    piano_roll = midi_data.get_piano_roll(fs=100)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(piano_roll, aspect='auto', origin='lower',
              cmap='plasma', interpolation='nearest')
    ax.set_title('Piano Roll')
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('MIDI Note Number')
    plt.show()