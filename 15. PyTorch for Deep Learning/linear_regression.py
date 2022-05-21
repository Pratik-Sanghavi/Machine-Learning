import numpy as np
import torch

# training data
# Input (temp, rainfall, humidity)
inputs = np.array([
    [73,67,43],
    [91, 134,58],
    [87,134,58],
    [102,43,37],
    [69,96,70]
], dtype = 'float32')

# Targets (apples, oranges)
targets = np.array(
    [
        [56,70],
        [81,101],
        [119,133],
        [22,37],
        [103,119]
    ], dtype = 'float32'
)

# Convert to tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

w = torch.randn(2,3, requires_grad=True)
b = torch.randn(2, requires_grad=True)

def model(x):
    # @ character implies matrix multiplication in torch
    # broadcasting happens to b to make dimensions equal
    return x @ w.t() + b

preds = model(inputs)

# Loss function
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff)/diff.numel()

loss = mse(preds, targets)
print(loss)

# Compute gradients
loss.backward()

# Gradients for weights
print(w.grad)
print(b.grad)

# reset grad values to zero
w.grad.zero_()
b.grad.zero_()
print(w.grad)
print(b.grad)

# Model training section
EPOCHS = 1000
for i in range(EPOCHS):
    preds = model(inputs)
    loss = mse(preds, targets)
    # if i!=EPOCHS-1:
    #     print(loss.detach().numpy(), end="->")
    # else:
    #     print(loss.detach().numpy())
    loss.backward()
    # Adjust weights and reset gradients
    # torch.no_grad() tells pytorch that I'm done with my gradient calculations; when I'm doing these operations don't track them for gradient work
    with torch.no_grad():
        w -=w.grad * 1e-5
        b -=b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()

print(model(inputs), targets)

# Using built-in functions
import torch.nn as nn

# Dataset and DataLoader
from torch.utils.data import TensorDataset
train_ds = TensorDataset(inputs, targets)
train_ds[0:3] # when we need a specific slice of data use TensorDataset

# DataLoader
from torch.utils.data import DataLoader

#define data loader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size=5, shuffle = True)

# Data loader is typically used in a for-in loop.
for xb, yb in train_dl:
    print(xb)
    print(yb)
    break

# nn.Linear
# Instead of initializing the weights and biases manually, we can define the model using the nn.Linear class from PyTorch

model = nn.Linear(3, 2)
print(model.weight)
print(model.bias)

# to get all the parameters of the model, use the parameters method
print(list(model.parameters()))

# Get the loss function from the library
import torch.nn.functional as F

# Define the loss function
loss_fn = F.mse_loss

loss = loss_fn(model(inputs), targets)
print(loss)

opt = torch.optim.SGD(model.parameters(), lr = 1e-5)

def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb,yb in train_dl:
            # 1. Generate Predictions
            pred = model(xb)
            # 2. Calculate the loss
            loss = loss_fn(pred, yb)
            # 3. Compute gradients
            loss.backward()
            # 4. Update parameters using gradients
            opt.step()
            # 5. Reset the gradients to zero
            opt.zero_grad()

            # Print the progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.2f}')

fit(100, model, loss_fn, opt)