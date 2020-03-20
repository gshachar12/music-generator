import py_midicsv
import numpy as np
import input_utils
import os
import pickle
from mido import Message, MidiFile, MidiTrack
from music21 import *
import random
from music import chord_notation, metronome, midi_scale, midi2note
from composer import sequence

n_x=128
"""
version 1.21- working- please don't change unless you have something really good to add

"""
#np.random.seed(0)
#random.seed(0)
def write2csv(midi, txt):
    # enter the information from the csv file to a text file
    with open (txt, "w")as file:
        for l in py_midicsv.midi_to_csv (midi):
            file.writelines (l + "\n")


# midi file to matrix encoder
# fits input from midi files to numpy 3d arrays
vocabulary_size = 130  # 0-127 are different pitches, 128 note off and 129 for nothing


class encoder:
    def __init__(self, midi):
        self.midi = midi  # create a text file that corresponds to the midi file
        self.csv = py_midicsv.midi_to_csv (midi)
        self.min=float()
        self.max=float()
        self.tempo = int (self.csv[0].split (', ')[-1][:-1])  ### tempo calculation *******

    def midi2matrix(self):

        note_events = [line for line in self.csv if "Note_" in line]  # remove unnecessary characters from the midi file

        notes_flag = [1 if parameter.split (", ")[2] == "Note_on_c" else 0 for parameter in note_events]
        durations = [int (parameter.split (", ")[1]) for parameter in
                     note_events]  # get note durations from note events

        pitches = [int (parameter.split (", ")[4]) for parameter in note_events]  # get the pitches from note events
        self.min=min(pitches)
        self.max=max(pitches)
        velocities = [int (parameter.split (", ")[5][:-1]) for parameter in
                      note_events]  # get the velocities from note events

        return input_utils.input2matrix (pitches, durations, velocities, notes_flag,self.min ,self.max )


# translates the given matrix/3d data type into an audible file
class decoder:
    def __init__(self, matrix, minimal_note, maximal_note):
        self.matrix = matrix
        self.min=minimal_note
        self.max=maximal_note
    def matrix2midi(self):
        matrix = self.matrix
        mid = MidiFile()  # create a new midi file
        mid.ticks_per_beat = 384  # tempo
        track = MidiTrack()  # build a new track
        mid.tracks.append(track)
        n, T = matrix.shape
        interval=int(matrix[n-1, 0])
        print (interval)
        # handle the first note event

        for pitch in range (n-1):

            velocity = int(matrix[pitch, 0])  # the current velocity of the pitch
            if velocity != 0:

                track.append(Message ('note_on', note=pitch+self.min, velocity=120, time=interval))
                interval=0
        for t in range (1, T):
            interval = int(matrix[n-1, t] - matrix[n-1, t - 1])
            for pitch in range (n-1):

                if matrix[pitch, t] != matrix[pitch, t - 1]:

                    velocity = int(matrix[pitch, t])  # the current velocity of the pitch


                    track.append(Message('note_on', note=pitch+self.min, velocity=(120 if velocity != 0 else 0),time=interval))
                    interval=0

                else:
                    continue

        mid.save ('c://project/music/new_song.mid')
        write2csv('c://project/music/new_song.mid', 'c://project/music/new_song.txt')

def generate( melody, piece_length,scale="c", bpm=60):
    piece = stream.Score()
    quarter_length = tempo.MetronomeMark(number=bpm) # the rhythm of the piece
    piece.append(quarter_length)
    degrees=sequence(piece_length)
    chord_progression = [chord_notation (scale, degree) for degree in degrees]
    def harmony_track(melody):
        harmonic_line = stream.Part()
        for i in range(piece_length):
            current_chord=chord_progression[i]
            d = duration.Duration ()
            d.quarterLength = 0.5*len(melody)
            harmonic_line.append(chord.Chord(current_chord,  duration=d)) ##### should be modified
        return harmonic_line
    def melody_track(melody):
        melodic_line=stream.Part()

        for i in range(piece_length):

            current_chord = chord.Chord (chord_progression[i]).pitchedCommonName
            print (current_chord)
            current_scale = ((current_chord[0] if current_chord[1]=="-" else current_chord[:2])   + ("m" if 'minor' in current_chord else "")).lower ()
            print (current_scale)
            scale_notes = midi_scale (current_scale)

            melody_notes = [scale_notes[melody[j] - 1] for j in range (len (melody))]
            print (melody)
            for j in range(len(melody)):

                melodic_line.append (note.Note (melody_notes[j], type='eighth'))


        return melodic_line

    harmony = harmony_track(melody)

    #piece.insert (melody_track(melody) )
    print (harmony)

    piece.insert (harmony)
    print (melody)

    piece.insert (melody_track(melody))
    s = midi.realtime.StreamPlayer(piece)
    while True:
        s.play()

