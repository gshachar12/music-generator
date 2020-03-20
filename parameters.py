"""parameters section"""
import numpy as np
import random

from suffix_trees import STree
from music import midi_scale, midi2note
# music parameters

minimal_note = 48
maximal_note = 60
quarter_length = 960
chords_notes = 3  # the number of allowed notes in a chord
scale = midi_scale("c")  # all the notes of the given scale
num_notes = len(scale)-1  # currently 7


dissonance = {1: 1.7,
              2: 0,
              3: 1.6,
              4: 0.1,
              5: 1.5,
              6: 0.1,
              7: 0.001
              }
# ga parameters

# -model parameters
maximum_epochs = 100
generations=50
elite=3 # number of best individuals
kill_percent = 0.25
# mutation parameters
ind_mutated = 0.2  # number of mutated individuals in the population
gene_mutated = 0.2  # of mutated genes in each vector
mutation_step = 0.05  # mutation coefficient

# -fitness parameters

#raters
"""Unique Rhythm Values"""
"""Crazy notes rater"""
# the crazy note will be defined as interval of more than one octave
# check that there aren't crazy notes- notes that are to far from each other

raters_coefficients = {'uniqueness':4.5,   #5,
                       'dissonance':2,
                       'pattern': 5,
                       'rhythm': 6
                       }  # coefficients that help to alter the effect of each rater

class composition:
    def __new__(cls, length):
        # the length of the composition

        composition=list(np.random.choice( range(1,num_notes), size=length)) # forming the composition's int representation

        return composition

class individual: # individual class

    def __init__(self, length):

        self.length=length
        # the composition that the individual holds
        self.composition=composition(length)

        self.num_gene_mutated = int(length *gene_mutated) # number of mutated genes
        # criteria that determine the quality of the composition ,
        self.fitness=float() # the information distance from the guides

    def cross_over(self, other, ratio=0.3 ):
        offspring=individual(self.length)# an offspring melody that contains the values from 2 parents
        new_comp=[]
        for i in range(len(other.composition)):
            percent = random.randint(0, 100) # equally distribute the genes picked with respect to the ratio
            if percent < ratio:  # add a gene from the first parent
                new_comp.append(self.composition[i])
            else:
                new_comp.append(other.composition[i])

        offspring.composition=new_comp
        return offspring

    def mutation(self):

        """
        mutate a given percent of the vectors in each composition

        :param ind_percent:
        :param gene_percent:
        :param population:
        :return:
        """
        # random note mutator-note pitch
        for i in range(self.num_gene_mutated):
            random_position=random.randint(0, len(self.composition)-1)
            random_note=random.randint(1, num_notes-1)
            self.composition[random_position]= random_note

    def calculate_fitness(self):
        # The fitness is a value between 0 and 1, which represents how close the matrix is to sound appealing.
        # this section is devided to subraters which help control the quality of the pitch

        # Neighboring pitch range
        raters=[]
        """dissonances and consonances rater"""

        dc_rater = np.sum([dissonance[note] for note in self.composition])*raters_coefficients['dissonance']
        raters.append(dc_rater)

        """ unique notes rater"""
        unique_elements=set(self.composition)
        u_rater=len(unique_elements)/float(len(self.composition))*raters_coefficients['uniqueness']

        raters.append(u_rater)

        """repeating patterns"""

        """direction stability-measures the number of times the composition changes directions"""

        """Repetition Rating"""

        """order of beats- the first and last elements of the list should be consonant"""

        """strong beats"""
        self.fitness=np.sum(raters)




def dc(list):
    return np.sum ([dissonance[note] for note in list])

#def longest_pattern(list):
#    pattern=""
#    st = STree.STree (''.join(str(note) for note in list)).lcs()

 #   print (st)  # [0, 8]
"""

def main():
    longest_pattern([1,2,3,1,2,3])

if __name__ == '__main__':
    main()
"""