
""" data representation of the input- converts the musical input into a machine readable language"""

import numpy as np
normalize_val = 100.0


""" parameters and initialization"""
n_h = 90  # the number of hidden states
n_x = 88  # the number of notes in the piano, the number of unique possible notes in the database
T_x = 24  # the number of time steps

def input2vector(lst):
    """
    :param lst: a list of numbers to be converted to a vector
    :return:
    """
    T_x=len(lst)
    one_hots=np.zeros((n_x, T_x))
    #build a one hot vector from the given input
    for t in range(T_x):

        one_hot = np.zeros((n_x))  # a new array of zeros
        one_hot[lst[t]] = 1

        one_hots[:,t]=one_hot
    return one_hots
def fit3d(input):
    x=np.array(n_x, m, T_x)

def main():
    print (input2vector([1,23,4]).shape)
if __name__ == '__main__':
    main()