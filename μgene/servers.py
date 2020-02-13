import numpy as np
from input_utils import dataset
from midi import decoder
import matplotlib.pyplot as plt
import pygame
"""parameters section"""

"""
ratio between dissonants and consonants"""
pitches=128
ind_mutated = 0.4# number of mutated individuals in the population
gene_mutated = 0.5# of mutated genes in each vector
kill_percent=0.25

np.random.seed(0)
guides = dataset("c://project/music/well tempered").matrices[:2]


def length(matrix):
    # return the duration of the piece
    return matrix.shape[1]

class composition:

    # the number of allowed pitches in the composition
    # the length of the composition
    def __new__(cls, pitches, length):
        composition = np.zeros ((pitches, length))
        composition[60:80,:] = 1
        np.random.shuffle (composition)

        return composition

class individual:
    def __init__(self, length):
        self.composition=composition(pitches, length)
        self.allowed_notes=np.zeros(pitches)
        # criteria that determine the quality of the composition ,
        self.distance=float() # the information distance from the guides

class population:
    def __init__(self, num_individuals, length):
        self.population = [individual(length) for _ in range(num_individuals)]

    def train(self):
        minimal_errors=[]
        generations=100

        for generation in range(generations):
            for individual in self.population:
                individual.distance=calculate_fitness(individual.composition, guides) # calculate the fitness function of the current individual

            self.population.sort(key=lambda x: x.distance, reverse=True) # sort the population by the distance at tribute of each of the individuals
            print ("generation: ", generation )
            minimal_error=self.population[0].distance
            print("current distance",minimal_error )
            minimal_errors.append(minimal_error)

            self.cross_over(kill_percent) # apply the cross over function
            self.population = mutation(ind_mutated, gene_mutated, self.population)
            generation+=1
        plot(minimal_errors, generations)
        final_matrix=self.population[0].composition

        final_matrix[0,:]=guides[0][0,:]
        print (final_matrix.shape)
        generated_midi=decoder(final_matrix).matrix2midi()
    def cross_over(self, kill_percent):
        k = 2  # number of crossover points

        kill_num = int (kill_percent * len (self.population))
        offsprings = []
        self.population = self.population[:-kill_num]  # kill the less fitting individuals

        # apply the k-point cross over algorithm
        for i in range (kill_num):
            parent1 = self.population[i]
            parent2 = self.population[i + 1]
            duration = length(parent1.composition)

            n1 = np.random.randint (1, duration / 2)
            n2 = np.random.randint (n1 + 1, duration)
            offspring = individual (duration)  # create off spring
            offspring.composition = parent2.composition
            offspring.composition[:, n1:n2] = parent1.composition[:, n1:n2]
            offsprings.append (offspring)
        self.population += offsprings

        """
        drawbacks:
        I am not sure that this is the best cross over method (i and i+1
        k-point cross over
        """

def stop_condition(iterations):
    maximum_epochs=400
    return iterations <= maximum_epochs


def calculate_fitness(composition, guides):
    # implement the ncd algorith

    longer_piece=lambda x,y: (x,y) if length(x)==max(length(x), length(y)) else (y,x)

    errors = [] # a vector that will contain all the errors. ########################

    for guide in guides:
        # find out which composition is longer, the guide or the individual
        # equalize the dimensions of the two matrices
        short,long=longer_piece(composition, guide)  ##############################

        long=np.resize(long, short.shape)  #one of the pieces is truncated-not sure about this step

        # function: arithmetic mean square error

        error=np.square(short-long).mean()
        errors.append(error)
    total_error=np.array(errors).mean() # calculate the mean value of the errors-the total error

    return total_error


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
    ind_mutated = int(ind_percent*len(population))  # number of mutated individuals in the population
    gene_mutated = int(gene_percent*len(population))  # number of mutated genes
    for i in range(len(population)):  # for each individual in the population
        composition = population[i].composition
        for _ in range(ind_mutated):
            for _ in range(gene_mutated):
                rand_vector=np.random.choice (composition.shape[1]) # pick a random vector from the matrix
                mutated_notes = composition[:, rand_vector]  # pick a random note vector from the matrix
                rand_note=np.random.choice (mutated_notes.shape[0])
                gene = mutated_notes[rand_note] # a note in the vector
                gene = 1 if gene == 0 else 0  # mutate one of the notes in the vector
                population[i].composition[:,rand_vector][rand_note]=gene # mutate

    """ current bugs:
            might mutate the same note twice
    """
    return population

def main():
    duration=length(guides[0])
    p=population(100,duration)
    p.train()


if __name__ == '__main__':

    main()
