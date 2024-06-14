import torch
import numpy as np
from torch import nn
import os

opacity = torch.tensor([1, 2, 3, 4, 5])
mask = torch.tensor([True, True, False, True, False])
new_opacity = opacity[mask].repeat(2)
print("new opacity : ", new_opacity)
dict = {"opacity": new_opacity, "mask": mask}
extension_tensor = dict["opacity"]
print("extension tensor : ", extension_tensor)
final  = torch.cat((opacity, extension_tensor), dim=0)
print("concatenated tensor : ", final)