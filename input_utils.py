
""" data representation of the input- converts the musical input into a machine readable language"""

import numpy as np
normalize_val = 100.0


""" parameters and initialization"""
n_h = 90  # the number of hidden states
n_x = 88  # the number of notes in the piano, the number of unique possible notes in the database
T_x = 24  # the number of time steps

def input2matrix(pitches, velocities, time_stamps, notes_flag):
    """

    :param pitches:
    :param velocities:
    :param time_stamps:
    :return:
    """
    T_x=time_stamps[-1]+1 # number of ticks
    n_x=128 # number of pitches in a vector (there are 128 possible pitches in a midi file)

    #build a one hot vector from the given input
    notes_vector = np.zeros((n_x))  # a new array of zeros that will be used to keep track of the notes being played

    matrix=np.zeros((n_x, T_x))
    for t in range(len(time_stamps)):
        pitch=pitches[t] # the current pitch being played

        velocity=velocities[t] # the current velocity of the pitch
        if notes_flag[t]: # a note is being played
            notes_vector[pitch]=velocity
        else: # note off event
            notes_vector[pitch] = 0
        matrix[:, time_stamps[t]] = notes_vector

    return matrix
def fit3d(input):
    x=np.array(n_x, m, T_x)

