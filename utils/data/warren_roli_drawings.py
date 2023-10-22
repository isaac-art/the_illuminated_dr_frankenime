
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, TensorDataset


def generate_diagonal_line(dim=100, width=5):
    img = Image.new('1', (dim, dim), 0)
    draw = ImageDraw.Draw(img)
    draw.line((0, 0) + (dim-1, dim-1), fill=1, width=width)
    return np.array(img)

def generate_L_shape(dim=100, width=5):
    img = Image.new('1', (dim, dim), 0)
    draw = ImageDraw.Draw(img)
    draw.line((10, 10, 10, dim-10), fill=1, width=width)
    draw.line((10, dim-10, dim-10, dim-10), fill=1, width=width)
    return np.array(img)

def generate_circle(dim=100, width=5):
    img = Image.new('1', (dim, dim), 0)
    draw = ImageDraw.Draw(img)
    draw.ellipse((10, 10, dim-10, dim-10), outline =1, width =width)
    return np.array(img)

def generate_s_shape(dim=100, font_size=100):
    img = Image.new('1', (dim, dim), 0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("_res/Arial.ttf", font_size)
    except IOError:
        print("Default font will be used.")
        font = ImageFont.load_default()
    text_width, text_height = draw.textsize('S', font)
    position = ((dim - text_width) // 2, (dim - text_height) // 2)
    draw.text(position, 'S', 1, font=font)
    return np.array(img)


def generate_data(num_samples=1000, dim=100):
    data = []
    labels = []
    
    for _ in range(num_samples):
        data.append(generate_diagonal_line(dim))
        labels.append(0)
        
        data.append(generate_L_shape(dim))
        labels.append(1)
        
        data.append(generate_circle(dim))
        labels.append(2)
        
        data.append(generate_s_shape(dim))
        labels.append(3)
    
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)
    
    return data, labels

def apply_transforms(dataset, data_transforms):
    transformed_data = []
    for sample in dataset:
        transformed_sample = data_transforms(sample)
        transformed_data.append(transformed_sample)
    return torch.stack(transformed_data)

def make_roli_dataset(output_path="datasets/drawing_data.pt"):
    data, labels = generate_data()
    print(f"Generated data shape: {data.shape}, labels shape: {labels.shape}")
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=(5, 5)),
        transforms.ToTensor(),
    ])
    data = data.unsqueeze(1) 
    transformed_data = apply_transforms(data, data_transforms)
    dataset = TensorDataset(transformed_data, labels)
    torch.save(dataset, output_path)
    print(f"Saved dataset to {output_path}")
    return dataset

def load_roli_dataset(dataset_path="datasets/drawing_data.pt"):
    dataset = torch.load(dataset_path)
    print(f"Loaded dataset from {dataset_path}")
    return dataset
