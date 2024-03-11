import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import List

from config import KernelConfig

@dataclass
class Gene():
    adj: List[bool]
    kernel_size: List[int]

    def __init__(self):
        self.adj = [np.random.choice([True, False]) for _ in range(6)]
        self.kernel_size = np.random.choice(KernelConfig.KERNEL_CHOICES, p=KernelConfig.SAMPLE_PROB, size=6).tolist()

    def get_matrix(self, i, j):
        if j == 2:
            if i == 1:
                return self.adj[0]
        if j == 3:
            if i <= 2:
                return self.adj[i]
        if j == 4:
            if i <= 3:
                return self.adj[i+2]
        return False
    
    def get_prev(self, i):
        return [j for j in range(1,7) if self.get_matrix(j, i)]
    def has_no_prev(self, i):
        return len(self.get_prev(i)) == 0
    
    def get_after(self, i):
        return [j for j in range(1,7) if self.get_matrix(i, j)]
    def has_no_after(self, i):
        return len(self.get_after(i)) == 0

class Individual():
    def __init__(self, gene1, gene2):
        self.stage1 = gene1
        self.stage2 = gene2
        self.evaluation = 0



class Node(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Node, self).__init__()
        self.seq = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2),
        )

    def forward(self, x):
        return self.seq(x)
    
class Block(nn.Module):
    def __init__(self, gene, in_channels, block_channels, out_channels):
        super(Block, self).__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.gene = gene

        self.node_intro = Node(in_channels, block_channels, 3)
        self.nodes = nn.ModuleList([Node(block_channels, block_channels, k) for k in gene.kernel_size])

        self.no_afters = [i for i in range(1,7) if self.gene.has_no_after(i)]
        self.out_node_idx = []
        for item in self.no_afters:
            if not self.gene.get_prev(item):
                self.out_node_idx.append(item)
        self.node_outro = Node(block_channels*len(self.out_node_idx), out_channels, 3)

    def forward(self, x):
        if self.gene.adj.count(True) == 0:
            return Node(self.in_ch, self.out_ch, 3)(x)
        
        x = self.node_intro(x)
        nodes_out =  []
        for idx in range(1,7):
            if self.gene.has_no_prev(idx):
                nodes_out.append(self.nodes[idx-1](x))
            else:
                temp = torch.zeros_like(x)
                for prev in self.gene.get_prev(idx):
                    temp += nodes_out[prev-1]
                nodes_out.append(self.nodes[idx-1](temp))

        cat_nodes_out = [(nodes_out[i-1]) for i in self.out_node_idx]
        outro_input = torch.cat(cat_nodes_out, dim=1)

        return self.node_outro(outro_input)
    
class Net(nn.Module):
    def __init__(self, individual, channel1, channel2):
        super(Net, self).__init__()
        self.block1 = Block(individual.stage1, 1, 8, 8)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.stage2 = Block(individual.stage2, 8, 16, 16)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.net = nn.Sequential(
            Block(individual.stage1, 1, channel1, channel1), # 28x28x1 -> 28x28xch1
            nn.MaxPool2d(2, 2), # 28x28xch1 -> 14x14xch1
            Block(individual.stage2, channel1, channel2, channel2),# 14x14xch1 -> 14x14xch2
            nn.MaxPool2d(2, 2), # 14x14xch2 -> 7x7xch2
            nn.Flatten(),
            nn.Linear(7*7*channel2, 200), nn.ReLU(),
            nn.Linear(200, 100), nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.net(x)
    
