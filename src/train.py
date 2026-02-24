import torch
from src.model import YOLOv1


m = YOLOv1(S=13, B=2, C=1)
x = torch.randn(4, 3, 416, 416)
y = m(x)
print(y.shape)  # torch.Size([4, 13, 13, 11])  (ker 2*5 + 1 = 11)
