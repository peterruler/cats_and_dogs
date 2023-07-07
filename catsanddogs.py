import torch
import torchvision
from torchvision import transforms
from PIL import Image

normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    normalize])
img = Image.open("catdog/train/cat.0.jpg")
img_tensor = transform(img) #(3,256,256)
img_tensor.unsqueeze_(0) #(1,3,256,256) _inplace wieder in img-tensor gespeichert 
print(img_tensor)