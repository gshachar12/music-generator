import numpy as np
from collections import Counter
from music import chord_notation
import pandas as pd
import matplotlib.pyplot as plt
import csv

file = "c://project/chord-progressions.csv"
# read chord progressions data from the database

file = csv.reader (open (file, 'r'))
chords = [[int(chord) for chord in row] for row in file]
used_chords = list (set (sum (chords, [])))  # unique elements in chords

def markov_chains():
    #guesses chords using the markov chain, and then sends the guess to the player
    #np.random.seed(2)  # save the current results
    bigrams = []  # create sequences of two notes

    for progression in chords:
        for i in range(len(progression)-1):
            bigrams.append((progression[i], progression[i+1])) # append tupples to the list- each
                                                                # one will be used to calculate the probability
                                                                # given the first one
    count_chords = dict((chord,bigrams.count(chord)) for chord in set(bigrams))  # returns the number of occurrences of a chord in the list
    occurrences = Counter([sequence[0] for sequence in bigrams])

    for key in count_chords:  # divide by the number of occurrences of the chord
        count_chords[key] = (float)(count_chords[key]) / occurrences[key[0]]
    size=len(used_chords)+1
    markov_matrix=np.zeros((size,size))

    markov_matrix[1:,0]=used_chords
    markov_matrix[0,1:]=used_chords
    for key, value in count_chords.items():

        rows, cols=key
        markov_matrix[used_chords.index(rows)+1, used_chords.index(cols)+1]=value
    return markov_matrix  # returns possible states in the system

def probability(current_chord):
     """

     :param first:
     :param second:
     :return: the probability for a second event given the first
     """
     file = "c://project/chord-progressions.csv"
     # read chord progressions data from the database

     file = csv.reader (open (file, 'r'))


     probability_matrix = markov_chains ()
     probabilities = probability_matrix[used_chords.index (current_chord)+1,1:]
     options = probability_matrix[1:, 0]
     next_chord = np.random.choice (options, p=probabilities)  # make a choice by following the given probabilities

     return next_chord


def main():
    print (probability(4))
if __name__ == '__main__':
    main()


