import torch
import torch.nn as nn

class SimpleLinearRegression(nn.Module):
    def __init__(self, input_f=1, output_f=1):
        super().__init__()

        self.fc = nn.Linear(input_f, output_f)
    
    def forward(self, x):
        return self.fc(x)

def training_loop(
    x_train, y_train,
    x_val, y_val,
    model, optimizer, criterion,
    epochs):

    for epoch in range(1, epochs+1):
        output_train = model(x_train)
        loss_train = criterion(output_train, y_train)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        with torch.no_grad():
            output_val = model(x_val)
            loss_val = criterion(output_val, y_val)