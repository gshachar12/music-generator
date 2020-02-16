import numpy as np
from input_utils import dataset
from midi import decoder
#import matplotlib.pyplot as plt
from lzma import compress
from mido import Message, MidiFile, MidiTrack
from music import midi_scale, midi2note
"""parameters section"""

"""
ratio between dissonants and consonants"""


# mutation parameters
ind_mutated = 0.4  # number of mutated individuals in the population
gene_mutated = 0.5  # of mutated genes in each vector
mutation_step=0.05
kill_percent=0.25

np.random.seed(0)
guides = dataset("c://project/music/test").matrices

minimal_note = 48
maximal_note = 60

quarter_length=960
chords_notes=3 # the number of allowed notes in a chord
scale=midi_scale("c") # all the notes of the given scale
maximum_epochs=100
class composition:


    def __new__(cls, length):
        # the length of the composition

        composition=np.random.choice(range(len(scale)), size=length) # forming the composition's int representation

        #composition_str = int2string()
        return composition

    def int2string(self):
        """transforms the int composition into string"""
        composition_str=""
        for event in self.composition_int:
            composition_str+="".join([midi2note(note) for note in event])+" "
        return composition_str

class individual:
    def __init__(self, length):

        self.length=length
        # the composition that the individual holds
        self.composition=composition(length)

        # criteria that determine the quality of the composition ,
        self.distance=float() # the information distance from the guides

    def crossover(self, other, ratio=0.5 ):
        offspring=individual(self.length)# an offspring melody that contains the values from 2 parents
        new_comp=[]
        for i in range(len(other_melody.composition)):
            percent = random.randint(0, 100) # equally distribute the genes picked with respect to the ratio
            if percent < ratio:  # add a gene from the first parent
                new_comp.append(self.composition[i])
            else:
                new_comp.append(other.composition[i])

        offspring.composition=new_comp
        return offspring

class population:
    def __init__(self, num_individuals, length):
        self.population = [individual(length) for _ in range(num_individuals)]

    def train(self):
        minimal_errors=[]
        generations=100

        for generation in range(generations):
            for individual in self.population:
                individual.distance=calculate_fitness(individual.composition, guides) # calculate the fitness function of the current individual

            self.population.sort(key=lambda x: x.fitness, reverse=True) # sort the population by the distance at tribute of each of the individuals
            print ("generation: ", generation )
            minimal_error=self.population[0].distance
            print("current distance",minimal_error )
            minimal_errors.append(minimal_error)

            self.cross_over(kill_percent) # apply the cross over function
            self.population = mutation(ind_mutated, gene_mutated, self.population)
            generation+=1
        #plot(minimal_errors, generations)
        final_melody=self.population[0].composition
        print (final_melody.shape)




def stop_condition(iterations):

    return iterations <= maximum_epochs


def calculate_fitness(composition, guides):
    # The fitness is a value between 0 and 1, which represents how close the matrix is to sound appealing.
    # this section is devided to subraters which help control the quality of the pitch

    # Neighboring pitch range
    # check that there aren't crazy notes- notes that are to far from each other
    # the crazy note will be defined as interval of more than one octave


    """direction stability-measures the number of times the composition changes directions"""
    ds_rate=[]/float(len(composition))

    """Equal Consecutive Notes"""
    """Unique Rhythm Values"""
    """dissonances and consonances"""
    # chord notes:


    dissonances=

def plot(minimal_errors, generations):
    # plot a graph showing the minimal error with respect to the current generation
    plt.plot ([i for i in range(generations)], minimal_errors)
    plt.xlabel ('genration')
    plt.ylabel ('minimal error')
    plt.show()


def mutation(ind_percent,gene_percent, population):

    """
    mutate a given percent of the vectors in each composition

    :param ind_percent:
    :param gene_percent:
    :param population:
    :return:
    """

    # finishes with tonic
    
    return population


def generate(length, melodic_pattern):
    melodic_pattern=[1, 1, 2, 3, 4, 4, 5,3]
    scale="cm"
def main():
    c=individual(10)
    print (c.composition)

if __name__ == '__main__':

    main()
