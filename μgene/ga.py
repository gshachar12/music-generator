import numpy as np
import string
import random
""" Genetic algorithm for reconstraction of a a target string """

target_string = "i believe that this system cannot understand this sentence at all   wow"
mutation_chance = 0.1
crossover_chance = 0.3
letters=string.letters+" "


def calc_fitness(target_string, current_string):
    """ calculates the fitness score of the given individual's string """
    fitness=0
    for i in range(len(target_string)):
        if target_string[i]==current_string[i]:
            fitness+=1
    return fitness

def rand_chromosome(num):
    # returns a random string with length of num characters
    return ''.join(random.choice(letters) for i in range(num))




class population:
    def __init__(self, num=100):
        self.population=[]
        self.num = num
        for i in range(num):
            self.population.append(individual(rand_chromosome(len(target_string)))) # random strings
    def get_fittness(self, index):
        return population[index].fittness
    def thanos(self,alive=0.08, mating=0.2):
        """ kill the unsuitable individuals"""
        self.population.sort(key=lambda x:x.fitness_score, reverse=True) # sort the pepulation by the fitness of each individual
        new_generation=self.population[:int(alive*len(self.population))]

        mating_group=int((1-alive)*len(self.population))
        mating=int(mating*len(self.population))
        for i in range(mating_group):
            # mate the other individuals
            parent1=self.population[random.randint(0,mating)]
            parent2 = self.population[random.randint(0, mating)]
            offspring=parent1.crossover(parent2)
            new_generation.append(individual(offspring))
        return new_generation

    def model(self):

        generation = 0  # the number of the current generation
        found = self.population[0].chromosome == target_string
        while not found:
            generation+=1
            self.population=self.thanos()

            print "generation: ", generation
            print "string: ", self.population[0].chromosome
            found = self.population[0].chromosome == target_string
class individual:

    def __init__(self, choromosome):
        self.chromosome=choromosome
        self.fitness_score=calc_fitness(target_string, choromosome)

    def crossover(self, parent2, mutation=0.2):
        offspring="" # an offspring string that contains the values from 2 parents

        p1 = (100 - mutation)/2.0  # percent for taking the genes from the first parent
        p2 = 2*p1  # percent for taking the genes from the second parent

        for i in range(len(parent2.chromosome)):
            percent = random.randint(0, 100)
            if percent < p1:  # add a gene from the first parent
                offspring += self.chromosome[i]
            elif percent < p2:
                offspring += parent2.chromosome[i]
            else:
                offspring+=rand_chromosome(1) # mutation
        return offspring

def main():
    p = population(25100).model()



if __name__ == '__main__':
    main()