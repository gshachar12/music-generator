import pyaudio
import numpy as np
from operator import itemgetter
from scipy import interpolate
import matplotlib.pyplot as plt

sampling_rate = 44100
maximal_midi_note=127
class Note:
    # piano-amplitude modulation
    notes = ['b', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#']  # list of possible notes
    base = 27.5  # the lowest key on the piano, which is A

    def __init__(self, note, octave):
        """arguments:
        note name
        the octave of the note (in the piano there are 7)"""

        self.note = note
        self.index = Note.notes.index(note)  # the index of the note in the above notes list
        self.frequency = Note.base * 2.0 ** (1.0 / 12.0 * self.index) * (2.0 ** octave)  # the frequency of the note

    def sine_wave(self, rate, length):
        return np.sin(2 * np.pi * np.arange(int(rate * length)) * float(self.frequency) / rate)


class music():
    """
     builds an music playing system- given a string of ascii characters,
     sound protocol:
        not spaced sting-chord
        numbers- the duration of the note (partial)"""

    def __init__(self, text=None, bpm=None, piece=None):

        if isinstance(piece, np.ndarray):
            self.piece = piece


        else:
            self.text = text  # text to convert
            self.bpm = bpm
            self.piece = self.unpack_text()

    def metronome(self):
        beat_duration = 60.0 / self.bpm
        return beat_duration

    def unpack_text(self):
        units = self.text.split(" ")  # units of sounds (chords and separate notes)
        piece = []

        for unit in units:
            notes = []
            if "/" in unit:
                unit, r = unit.split("/", 1)

            else:
                r = '1'

            for note in list(unit):  # create a list of notes names
                if note == "#":  # sharp (half tone above)
                    notes[-1] += note

                elif note.isdigit():
                    notes[-1] = (notes[-1], int(note))
                else:
                    notes.append(note)

            for i in range(len(notes)):

                if type(notes[i]) is not tuple:
                    notes[i] = (notes[i], 4)  # the default is the fourth octave

            notes = [Note(note[0], note[1]) for note in notes]  # create notes objects
            if "/" in r:  # turn the fraction into decimal
                r = float(r.split("/")[0]) / float(r.split("/")[1])
            else:
                r = float(r)
            duration = self.metronome() * float(r)
            piece.append(self.chord(notes, sampling_rate, duration) * 0.2)  # add all the chords into the final piece

        piece = np.concatenate(np.column_stack((piece)))
        return piece

    def __add__(self, other):
        """adds two music instances (playing multiple pieces simultaneously, i.e, a mash-up)"""

        max_shape = max(self.piece.shape, other.piece.shape)[0]  # maximal duration

        zeros = np.zeros(max_shape - self.piece.shape[0])
        self_piece = np.hstack((self.piece, zeros))
        zeros = np.zeros(max_shape - other.piece.shape[0])
        other_piece = np.hstack((other.piece, zeros))
        _sum = self_piece + other_piece

        return music(None, None, _sum)

    def chord(self, chunk, rate, length):
        """ Arguments:
        a list of notes to be played as a chord, rate and length of note.
        Outputs:
        a numpy nd array of the chords (piano sounds)"""
        chord = []
        adsr = {0.0: 0.00, 0.003: 3, 0.1: 2, 0.8: 0.2, 1.0: 0.0}  # first point, attack, decay, sustain, release
        for note in chunk:
            chord.append(self.ADSR(note.sine_wave(rate, length), adsr))
        chord = np.array(chord)
        chord = chord.sum(axis=0, keepdims=True)
        return chord * 0.34

    def ADSR(self, note, adsr, kind='slinear', display=False):

        # Attack, Decay, Sustain, Release
        items = adsr.items()  # get x and y values of the ADSR function
        items.sort(key=itemgetter(0))  # arrange x values (from the minimal to the maximal
        x = map(itemgetter(0), items)
        y = map(itemgetter(1), items)
        # predict the linear adsr function from the points
        adsr = interpolate.interp1d(x, y, kind=kind)
        new_sound = adsr(np.arange(len(note)) / float(len(note)))  # apply the adsr function on the given note. \
        # The maximal value is 1, so we have to divide by len(note).

        if display:
            plt.plot(x, y, 'o', (np.arange(len(note)) / float(len(note))), new_sound, '-')
            plt.show()
        return new_sound * note  # apply the function on the sound


def chord_notation(chord):
    "returns a string according to the chord notation"
    chord=chord.lower()

    major_intervals=[2,2,1,2,2,2,1]
    minor_scale_intervals = [2, 1, 2, 2, 1, 2, 2]
    intervals = minor_scale_intervals if "m" in chord else major_intervals # minor_chord if there is m
    chord_intervals = [3, 4] if "sus4" in chord else [2, 4] # intervals between the notes in the chord
    chord=chord[0]
    first=Note.notes.index(chord)



    chord_notation=[first+sum(intervals[:chord_intervals[0]]), first+sum(intervals[:chord_intervals[1]])]
    for i in range(len(chord_notation)):
        if chord_notation[i]>=len(Note.notes): # if the note is higher than possible
            chord_notation[i]=divmod(chord_notation[i],len(Note.notes))[1] # adjust the note to the scale
            chord_notation[i]= Note.notes[chord_notation[i]]+"5"
        else:
            chord_notation[i] = Note.notes[chord_notation[i]]
    return chord+chord_notation[0]+chord_notation[1]


def play(music_obj):
    chunk = music_obj.piece
    play = pyaudio.PyAudio()
    # sound stream
    stream = play.open(format=pyaudio.paFloat32, channels=1, rate=44100, output=True)
    stream.write(chunk.astype(np.float32).tostring())
    stream.close()
    play.terminate()

def circulation(lst):
    while True:
        # iterate circularly through a list. yield the next element each time
        for var in lst:
            yield var


def midi_scale(reference_note):
    # returns a vector representing the midi notes allowed for the scale
    major_intervals = [2, 2, 1, 2, 2, 2, 1]
    minor_scale_intervals = [2, 1, 2, 2, 1, 2, 2]
    intervals = minor_scale_intervals if "m" in reference_note else major_intervals  # minor_chord if there is m
    reference_note = reference_note[0]
    scale_notes = np.zeros((maximal_midi_note))  # a vector that contain will contain all the allowed notes in a scale
    current_note = notes2midi(reference_note+"0")


    intervals = circulation(intervals) # define a generator

    while current_note < maximal_midi_note: # fill the notes vector with the scale notes.

        # convert the note to the midi suitable number and turn on the suitable pitch in the vector
        scale_notes[current_note] = 1
        print (current_note)
        interval=next(intervals)
        current_note += interval # add the next interval to the current note
    return scale_notes[21:109] # trim the scale to piano notes

def notes2midi(reference_note):
    # returns the midi number of the given note. for example c3-->48
    key = reference_note[:-1]

    octave = int(reference_note[-1])
    note_index=Note.notes.index(key)

    midi_number = octave*12+note_index-1
    return midi_number



def main():
    print (notes2midi("c0"))
    print (midi_scale("a"))
if __name__ == '__main__':

    main()