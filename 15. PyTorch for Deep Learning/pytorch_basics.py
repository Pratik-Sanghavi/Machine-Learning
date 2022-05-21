import torch

# At its core, Pytorch is a library for processing tensors. 
# A tensor is a number, vector, matrix or any n-dimensional array. 
# Lets create a tensor with a single number

t1 = torch.tensor(4.)

#tensor(4.)
#torch.float32
print(t1)
print(t1.dtype)

# Vector
t2 = torch.tensor([1.,2,3,4]) # must be same datatype; if different converted to same datatype
print(t2)

#Matrix
t3 = torch.tensor([[5.,6],[7,8],[9,10]])
print(t3, t3.shape)

# 3-d Array
t4 = torch.tensor([
    [[11,12,13],
    [13,14,15]],
    [[15,16,17],
    [17,18,19]]
])
print(t4)

# Tensor operations and gradients
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad = True)
b = torch.tensor(5., requires_grad = True)

# Arithmetic operations
y = w*x + b
print(y)

# compute derivatives
print(y.backward())

print(f"dy/dx:{x.grad}")
print(f"dy/dw:{w.grad}")
print(f"dy/db:{b.grad}")

# interoperability with numpy

import numpy as np

x = np.array([[1,2],[3,4.]])

# Convert numpy array to pytorch tensor
y = torch.from_numpy(x)
# OR
y = torch.tensor(x)

print(x.dtype, y.dtype)

# Convert torch tensor to numpy array
z = y.numpy()
print(z)