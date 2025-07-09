import torch.nn as nn

class MLPRegression(nn.Module):
    def __init__(self, input_f=1, hidden_size=20, output_f=1):
        super().__init__()
        self.layer1 = nn.Linear(input_f, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_f)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

def training_loop_mlp(
    x_train, y_train,
    model, optimizer, criterion,
    epochs):

    for _ in range(1, epochs + 1):
        output_train = model(x_train)
        loss_train = criterion(output_train, y_train)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
    return loss_train.item()