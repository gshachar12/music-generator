""" data representation of the input- converts the musical input into a machine readable language"""

import numpy as np
normalize_val = 100.0


""" parameters and initialization"""
n_h = 90  # the number of hidden states

def input2matrix(pitches, durations, velocities, notes_flag):
    """
    :param pitches:
    :param velocities:
    :return:
    """
    n_x=128 # number of pitches in a vector (there are 128 possible pitches in a midi file)
    durations.sort() # sort the durations list
    events_time=sorted(set(durations))
    T= len(events_time)# the length of the input time sequences
    #build a one hot vector from the given input

    matrix=np.zeros((n_x, T))
    # repetition count setting
    repetition_count=np.array(events_time)
    matrix[0,:] = repetition_count
    notes_vector = np.zeros((n_x-1))  # a new array of zeros that will be used to keep track of the notes being played
    for t in range(len(durations)):
        pitch = pitches[t]  # the current pitch being played
        velocity =1# velocities[t]  # the current velocity of the pitch
        notes_vector[pitch-1] = velocity * notes_flag[t]

        matrix[1:, events_time.index(durations[t])] = notes_vector

    return matrix

def create_inputs(matrix, T_x, T_y):

    # seperate the inputs into ys and xs using the windowing method

    x_vals=np.zeros(n_x ,T_x, m) # these values will be fed as inputs to the lstm
    y_vals = np.zeros(n_x ,T_y, m)  # these values will be fed as outputs to the lstm

    # iterate through all the time steps in the matrix
    T= matrix.shape[1] # the length of the piece
    for t in range(T-max(T_x, T_y)):
        x_vals=matrix[t:T_x+t]

        y_vals = matrix[t:T_y + t]

    return x_vals, y_vals




def main():
    print (input2vector([1,23,4]).shape)
if __name__ == '__main__':
    main()