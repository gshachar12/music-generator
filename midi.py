import py_midicsv
import numpy as np
import input_utils
import os
from mido import Message, MidiFile, MidiTrack

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

       self.tempo = int(self.csv[0].split(', ')[-1][:-1]) ### tempo calculation *******

   def midi2matrix(self):

       note_events=[line for line in self.csv if "Note_" in line ] # remove unnecessary characters from the midi file

       notes_flag=[1 if parameter.split(", ")[2]=="Note_on_c" else 0 for parameter in note_events]
       durations=[int(parameter.split(", ")[1]) for parameter in note_events] # get note durations from note events
       pitches=[int(parameter.split(", ")[4]) for parameter in note_events] # get the pitches from note events
       velocities=[int(parameter.split(", ")[5][:-1]) for parameter in note_events] # get the velocities from note events
       return input_utils.input2matrix(pitches,durations,  velocities, notes_flag)


# translates the given matrix/3d data type into an audible file
class decoder:
    def __init__(self, matrix):
        self.matrix=matrix

    def matrix2midi(self):
        matrix=self.matrix
        mid = MidiFile() # create a new midi file
        mid.ticks_per_beat=200 # tempo
        track = MidiTrack() # build a new track
        mid.tracks.append(track)
        n_x, T_x=matrix.shape

        prev_t=0
        # handle the first note event
        for pitch in range(1, len(matrix[:, 0])):

            velocity = int(matrix[pitch, 0])  # the current velocity of the pitch
            if velocity!= 0:

                track.append(Message('note_on', note=pitch, velocity=velocity, time=0))

        for t in range(1,T_x):
            count=0 # handling delta t
            flag=1
            for pitch in range(1, n_x):

                if matrix[pitch,t]==matrix[pitch,t-1]:
                    continue
                velocity = int(matrix[pitch, t])  # the current velocity of the pitch
                if count:
                    flag=0
                track.append(Message('note_on', note=pitch, velocity=velocity, time=flag*int(matrix[0,t]-matrix[0,t-1])))
                count+=1



        for pitch in range(len(matrix[:, prev_t])):

            velocity = int(matrix[pitch, prev_t])  # the current velocity of the pitch
            if velocity != 0:
                track.append(Message('note_off', note=pitch, velocity=0, time= int(T_x -1- prev_t)))
        mid.save('c://project/music/new_song.mid')

def create_dataset(directory):
    """

    :param directory: the name of the directory
    :return:
    creates a dataset from the files of the directory
    """
    write2csv('c://project/music/test/amazing.midi','c://project/csv.txt')
    write2csv('c://project/music/new_song.mid', 'c://project/midi.txt')
    dataset=list()
    files=os.listdir(directory)
    for file in files:
        print (file)
        dataset.append(encoder(directory+"/"+file).midi2matrix())
        print ("interpreted:", 100*((files.index(file)+1)/float(len(files))),"%")
    return dataset
def main():

    create_dataset("c://project/music/test")
    e=decoder(create_dataset("c://project/music/test")[2]).matrix2midi()
if __name__ == '__main__':
    main()