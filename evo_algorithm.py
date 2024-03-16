import numpy as np
import torch
import gc

from train_test import train, test
from model import Gene, Individual, Net

from config import TrainConfig, KernelConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def flip_coin(prob):
    x = np.random.choice(range(2),p=[1-prob,prob])
    return x==1

def exchange(list1, list2, idx):
    temp = list1[idx:]
    list1[idx:] = list2[idx:]
    list2[idx:] = temp


class Fauna():
    def __init__(self, size, channel1, channel2, mutation_prob, crossover_prob):
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.train_epochs = TrainConfig.TRAIN_EPOCHS
        self.channel1 = channel1
        self.channel2 = channel2
        self.size = size
        self.individuals = [Individual(Gene(), Gene()) for _ in range(size)]
        self.avg_eval = 0
        self.max_eval = 0
        self.min_eval = 1000
        self.max_individual = None
    
    def evaluate(self, train_loader, test_loader):
        for individual in self.individuals:
            model = Net(individual, self.channel1, self.channel2).to(device)
            train(model, train_loader, self.train_epochs, TrainConfig.LEARNING_RATE, device)
            correct_rate = test(model, test_loader, device)
            individual.evaluation = correct_rate
            if correct_rate > self.max_eval:
                self.max_eval = correct_rate
            if self.max_individual is None or correct_rate > self.max_individual.evaluation:
                self.max_individual = individual
            if correct_rate < self.min_eval:
                self.min_eval = correct_rate
            self.avg_eval += correct_rate
            
            del model  
            torch.cuda.empty_cache()  
            gc.collect() 
        self.avg_eval /= self.size
    
    def select_russian(self):
        difference = [individual.evaluation - self.min_eval for individual in self.individuals]
        softmax = [np.exp(d) for d in difference]
        probs = [d/sum(softmax) for d in softmax]
        new_individuals = np.random.choice(self.individuals,p=probs,size=self.size).tolist()
        self.individuals = new_individuals
        self.avg_eval = 0
        self.max_eval = 0
        self.min_eval = 1000
    
    def select_tournament(self):
        new_individuals = []
        for _ in range(self.size):
            sample1, sample2 = np.random.choice(self.individuals, size=2, replace=False)
            if sample1.evaluation > sample2.evaluation:
                new_individuals.append(sample1)
            else:
                new_individuals.append(sample2)
        self.individuals = new_individuals
        self.avg_eval = 0
        self.max_eval = 0
        self.min_eval = 1000

    
    def mutation(self):
        for individual in self.individuals:
            for gene in [individual.stage1, individual.stage2]:
                for i in range(6):
                    if flip_coin(self.mutation_prob):
                        gene.adj[i] = not gene.adj[i]
                for i in range(4):
                    if flip_coin(self.mutation_prob):
                        gene.kernel_size[i] = np.random.choice(KernelConfig.KERNEL_CHOICES, p=KernelConfig.SAMPLE_PROB)

    def crossover(self):
        for i in range(0, self.size, 2):
            if flip_coin(self.crossover_prob):
                idx = np.random.choice(range(6))
                exchange(self.individuals[i].stage1.adj, self.individuals[i+1].stage1.adj, idx)
            if flip_coin(self.crossover_prob):
                idx = np.random.choice(range(6))
                exchange(self.individuals[i].stage2.adj, self.individuals[i+1].stage2.adj, idx)
            if flip_coin(self.crossover_prob):
                idx = np.random.choice(range(4))
                exchange(self.individuals[i].stage1.kernel_size, self.individuals[i+1].stage1.kernel_size, idx)
            if flip_coin(self.crossover_prob):
                idx = np.random.choice(range(4))
                exchange(self.individuals[i].stage2.kernel_size, self.individuals[i+1].stage2.kernel_size, idx)
                
