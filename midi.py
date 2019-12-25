import py_midicsv
import numpy as np
import input_utils

from mido import Message, MidiFile, MidiTrack, MetaMessage

def write2csv(midi,txt):

    # enter the information from the csv file to a text file
    with open(txt, "w")as file:
        for l in py_midicsv.midi_to_csv(midi):
            file.writelines(l + "\n")

# midi file to matrix encoder
# fits input from midi files to numpy 3d arrays
vocabulary_size=130 # 0-127 are different pitches, 128 note off and 129 for nothing
class encoder:
   def __init__(self, midi):
       self.midi = midi # create a text file that corresponds to the midi file
       self.csv=py_midicsv.midi_to_csv(midi)
       print (self.csv)
       self.tempo = int(self.csv[0].split(', ')[-1][:-1]) ### tempo calculation *******

   def midi2matrix(self):

       note_events=[line for line in self.csv if "Note_" in line ] # remove unnecessary characters from the midi file

       notes_flag=[1 if parameter.split(", ")[2]=="Note_on_c" else 0 for parameter in note_events]
       durations=[int(parameter.split(", ")[1]) for parameter in note_events] # get note durations from note events
       pitches=[int(parameter.split(", ")[4]) for parameter in note_events] # get the pitches from note events
       velocities=[int(parameter.split(", ")[5][:-1]) for parameter in note_events] # get the velocities from note events
       return input_utils.input2matrix(pitches, velocities, durations, notes_flag)


# translates the given matrix/3d data type into an audible file
class decoder:
    def __init__(self, matrix, tempo):
        self.matrix=matrix
        self.tempo=tempo
    def matrix2midi(self):
        matrix=self.matrix
        mid = MidiFile() # create a new midi file
        mid.ticks_per_beat=self.tempo
        track = MidiTrack() # build a new track
        mid.tracks.append(track)
        n_x, T_x=matrix.shape

        prev_t=0
        g=0
        # handle the first note event
        for pitch in range(len(matrix[:,0])):

            velocity = int(matrix[pitch, 0])  # the current velocity of the pitch
            if velocity!= 0:
                track.append(Message('note_on', note=pitch, velocity=velocity, time=0))

        for t in range(1, T_x):
            # insignificant time stamps
            if not np.any(matrix[:, t]):  # if the array is an array of zeros
                continue
            else:
                count=0
                flag=1
                for pitch in range(len(matrix[:,t])):
                    if matrix[pitch,t]==matrix[pitch, prev_t]:
                        continue
                    velocity = int(matrix[pitch, t])  # the current velocity of the pitch
                    if count:
                        flag=0
                    if velocity != 0 :

                        track.append(Message('note_on', note=pitch, velocity=velocity, time=flag*int(t-prev_t)))
                        count+=1
                    elif matrix[pitch, prev_t]!=0:
                        track.append(Message('note_off', note=pitch, velocity=0, time=flag*int(t-prev_t)))
                        count+=1
                prev_t = t


        for pitch in range(len(matrix[:, prev_t])):

            velocity = int(matrix[pitch, prev_t])  # the current velocity of the pitch
            if velocity != 0:
                track.append(Message('note_off', note=pitch, velocity=0, time= int(T_x -1- prev_t)))
        mid.save('c://project/music/new_song.mid')

def main():

    m = encoder("c://project/music/amazing.midi")#.midi2matrix()
    matrix=m.midi2matrix()
    m=decoder(matrix, m.tempo).matrix2midi()
    write2csv("c://project/music/amazing.midi", "c:/project/mid.txt")
    write2csv("c://project/music/new_song.mid", "c:/project/csv.txt")
if __name__ == '__main__':
    main()