""" data representation of the input- converts the musical input into a machine readable language"""

import numpy as np

n_h = 90  # the number of hidden states
"""
version 1.21- working- please don't change unless you have something really good to add

"""
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

        velocity = velocities[t]  # the current velocity of the pitch
        notes_vector[pitch-1] = notes_flag[t]*velocity # the note is turned on or off


        matrix[1:, events_time.index(durations[t])] = notes_vector # cover the pitch section with the new vector
    print(matrix[:, 3] == matrix[:, 5])
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



