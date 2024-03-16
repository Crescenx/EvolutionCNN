class DataLoaderConfig:
    BATCH_SIZE = 64

class KernelConfig:
    KERNEL_CHOICES = [1, 3, 5, 7]
    SAMPLE_PROB = [0.15, 0.4, 0.3, 0.15]

class NetConfig:
    CHANNEL_1 = 8
    CHANNEL_2 = 16


class TrainConfig:
    LEARNING_RATE = 0.002
    TRAIN_EPOCHS = 10

class EvoAlgorithmConfig:
    MUTATION_PROB = 0.015
    CROSSOVER_PROB = 0.1
    POPULATION_SIZE = 30
    EVOLUTION_ROUNDS_1 = 15
    EVOLUTION_ROUNDS_2 = 10
