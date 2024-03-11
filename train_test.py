import torch.nn as nn
import torch

def train_epoch(model,dataloader,creterion = nn.CrossEntropyLoss(),learning_rate=0.001,device='cuda'):
    model.train()
    train_loss = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for data, lable in dataloader:
        data, label = data.to(device), label.to(device)
        output = model(data)
        loss = creterion(output, lable)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(dataloader)

def test(model, dataloader, device='cuda'):
    model.eval()
    correct_cnt = 0
    total_cnt = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_cnt += labels.size(0)
            correct_cnt += (predicted == labels).sum().item()
    return correct_cnt/total_cnt*1000 

def init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


def train(model, train_loader, epochs, learning_rate=0.001, device='cuda'):
    model = model.to(device)
    init(model)
    for _ in range(epochs):
        train_epoch(model, train_loader, learning_rate=learning_rate, device=device) 