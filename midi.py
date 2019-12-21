import py_midicsv
import numpy as np
import input_utils

# fits input from midi files to numpy 3d arrays
vocabulary_size=130 # 0-127 are different pitches, 128 note off and 129 for nothing
class fitting_input:
   def __init__(self, midi):
       self.csv = py_midicsv.midi_to_csv(midi) # create a text file that corresponds to the midi file
   def get_csv(self):
       # enter the information from the csv file to a text file
       two=0
       three=0
       with open("c:/project/mid.txt", "w" )as file:
           for l in self.csv:
               if l[0]=='2':
                print (l)
                two+=1
               if l[0]=='3':
                three+=1
               file.writelines(l+"\n")
           print (three,two)
   def midi2numpy(self):
       float_lst=[float(line.replace("Note_on_c, ", "").split(", ")[3]) for line in self.csv if "Note_on_c" in line ] # remove unnecessary characters from the midi file

       input = input_utils.input2vector(float_lst) # creating a numpy array of the frequencies+ normalization
       return input
m=fitting_input("c://project/music/amazing.midi").get_csv()



