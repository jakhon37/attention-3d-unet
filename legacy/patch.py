import torch

S = 128 # channel dim
W = 256 # width
H = 256 # height
batch_size = 10

x = torch.randn(batch_size, S, W, H)

size = 64 # patch size
stride = 64 # patch stride
patches = x.unfold(1, size, stride).unfold(2, size, stride).unfold(3, size, stride)
print(patches.shape)