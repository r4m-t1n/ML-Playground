import torch.nn as nn

class SimpleLinearRegression(nn.Module):
    def __init__(self, input_f=1, output_f=1):
        super().__init__()

        self.fc = nn.Linear(input_f, output_f)
    
    def forward(self, x):
        return self.fc(x)

def training_loop_slr(
    x_train, y_train,
    model, optimizer, criterion,
    epochs):

    for _ in range(1, epochs+1):
        output_train = model(x_train)
        loss_train = criterion(output_train, y_train)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
    return loss_train.item()