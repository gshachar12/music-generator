import py_midicsv
import numpy as np
import input_utils

#https://esc.fnwi.uva.nl/thesis/centraal/files/f1647680373.pdf

# midi file to matrix encoder
# fits input from midi files to numpy 3d arrays
vocabulary_size=130 # 0-127 are different pitches, 128 note off and 129 for nothing
class encoder:
   def __init__(self, midi):
       self.csv = py_midicsv.midi_to_csv(midi) # create a text file that corresponds to the midi file
   def get_csv(self):
       # enter the information from the csv file to a text file
       with open("c:/project/mid.txt", "w" )as file:
           for l in self.csv:
               file.writelines(l+"\n")

   def midi2numpy(self):
       note_events=[line.replace("Note_on_c, ", "").split(", ") for line in self.csv if "Note_on_c" in line ] # remove unnecessary characters from the midi file
       durations=[parameter[0] for parameter in note_events] # get note durations from note events
       pitches=[parameter[3] for parameter in note_events] # get the pitches from note events
       pitches=[parameter[3] for parameter in note_events] # get the velocities from note events
       input = input_utils.input2vector(float_lst) # creating a numpy array of the frequencies+ normalization
       return input
m=encoder("c://project/music/simpson.mid").get_csv()


# translates the given matrix/3d data type into an audible file
class decoder:

    def __init__(self, matrix):
        self.csv = py_midicsv.midi_to_csv (midi)  # create a text file that corresponds to the midi file

    def get_csv(self):
        # enter the information from the csv file to a text file
        with open ("c:/project/mid.txt", "w")as file:
            for l in self.csv:
                file.writelines (l + "\n")

    def midi2numpy(self):
        float_lst = [float (line.replace ("Note_on_c, ", "").split (", ")[3]) for line in self.csv if
                     "Note_on_c" in line]  # remove unnecessary characters from the midi file

        input = input_utils.input2vector (float_lst)  # creating a numpy array of the frequencies+ normalization
        return input