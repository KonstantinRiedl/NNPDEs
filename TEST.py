import os, datetime, inspect, random
import json
import pickle

import numpy as np

import torch
import torch.nn.functional as F


print("Hello World")

print(torch.cuda.is_available())

# GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

a = torch.rand(2, 2, 3)
print(a)
a.to(device)
print(a.device)
print(a)
print(a.get_device())
print(a.cpu().get_device())


b = torch.rand(2, 2, 3)
print(b)
b = b.to(device)
print(b.device)
print(b)
print(b.get_device())
print(b.cpu().get_device())


c = torch.rand(2, 2, 3, device=device)
print(c)
print(c.device)
print(c)
print(c.get_device())
print(c.cpu().get_device())

