import os
from mido import MidiFile, MidiTrack, merge_tracks
from collections import defaultdict

input_dir = './output/mss/omnizart'
output_dir = './output/mss/omnizart/combined'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

files_by_group = defaultdict(list)
for filename in os.listdir(input_dir):
    if filename.endswith('.mid'):
        prefix = filename.split('_')[0]
        files_by_group[prefix].append(filename)

for group, filenames in files_by_group.items():
    tracks = []
    for filename in filenames:
        mid = MidiFile(os.path.join(input_dir, filename))
        for i, track in enumerate(mid.tracks):
            tracks.append(track)
    
    combined_mid = MidiFile()
    combined_track = MidiTrack()
    combined_mid.tracks.append(combined_track)
    for msg in merge_tracks(tracks):
        combined_track.append(msg)
    
    output_filename = f'{group}_combined.mid'
    combined_mid.save(os.path.join(output_dir, output_filename))
    print(f'Combined {filenames} into {output_filename}')
