
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
        velocity = velocities[t]  # the current velocity of the pitch
        notes_vector[pitch-1] = velocity * notes_flag[t]

        matrix[1:, events_time.index(durations[t])] = notes_vector

    return matrix

