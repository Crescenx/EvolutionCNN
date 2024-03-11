class DataLoaderConfig:
    BATCH_SIZE = 64

class KernelConfig:
    KERNEL_CHOICES = [1, 3, 5, 7]
    SAMPLE_PROB = [0.1, 0.4, 0.4, 0.1]

class NetConfig:
    CHANNEL_1 = 8
    CHANNEL_2 = 16


class TrainConfig:
    LEARNING_RATE = 0.001
    TRAIN_EPOCHS = 5

class EvoAlgorithmConfig:
    MUTATION_PROB = 0.05
    CROSSOVER_PROB = 0.25
    POPULATION_SIZE = 2
    EVOLUTION_ROUNDS = 2
