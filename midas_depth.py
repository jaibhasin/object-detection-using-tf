import torch

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to("cpu").eval()

transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
