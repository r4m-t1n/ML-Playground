import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_f, hidden_size, output_f):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_f, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_f),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def training_loop_mlp_classifier(
    x_train, y_train,
    model, optimizer, criterion,
    epochs):

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    return loss.item()