import numpy as np
from input_utils import dataset
import random
import matplotlib.pyplot as plt
from midi import generate
from music import midi_scale, midi2note
from music21 import *
from parameters import *

import sys

#np.random.seed(0)
#random.seed(0)
class population:
    def __init__(self, num_individuals=32, length=8, plot_generations=False):# mutation parameters


        self.plot_generations=plot_generations
        self.num_individuals=num_individuals
        self.num_mutated=int(ind_mutated*num_individuals)


        # killing parameters

        self.num_kill=int(kill_percent*num_individuals)

        # cross-over parameters


        #number of points crossover
        # number of crossed parents

        # initialize population
        self.population = [individual(length) for _ in range(num_individuals)]

    def train(self):
        minimal_errors=[]

        for generation in range(generations):
            random.shuffle(self.population) # randomize the population
            for i in range(self.num_mutated):  # mutation
                self.population[i].mutation()

            for individual in self.population:
                individual.calculate_fitness() # calculate the fitness function of the current individual

            self.population.sort(key=lambda x: x.fitness, reverse=True) # sort the population by the distance at tribute of each of the individuals
            self.population=self.population[:-self.num_kill] # truncate the population

            minimal_error = dc(self.population[0].composition) # the fitness of the best individual

            print ("generation:", generation)
            print("current distance:",minimal_error )
            print ("current melody:",self.population[0].composition )


            minimal_errors.append(minimal_error)

            for i in range(self.num_kill):  # cross over
                parent1, parent2=np.random.choice(self.population[:elite], size=2)
                offspring=parent1.cross_over(parent2) # apply the cross over function
                self.population.append(offspring)


            #self.decrease_mutation()
        if self.plot_generations:
            plot(minimal_errors, generations)
        final_melodies = [individual.composition for individual in self.population[:elite]]
        return final_melodies
    def decrease_mutation(self):
        self.ind_mutated *= self.mutation_step
        self.num_mutated = int (self.ind_mutated *self.num_individuals )

def stop_condition(iterations):

    return iterations <= maximum_epochs



def plot(minimal_errors, generations):
    # plot a graph showing the minimal error with respect to the current generation
    plt.plot([i for i in range(generations)], minimal_errors)
    plt.xlabel('genration')
    plt.ylabel('minimal error')
    plt.show()

def main():
    print (sys.argv)
    bpm, scale=sys.argv[3], sys.argv[4]
    print (bpm, scale)
    bpm=int(bpm )
    melody=population(100, 6).train()[0] # length of each melody
    #bpm=100
    #scale="fm"

    #melody=[5, 3, 1, 3, 5, 3, 1,3] # predefined melody
    generate (melody, 8, bpm=bpm,  scale=scale)
if __name__ == '__main__':

    main()