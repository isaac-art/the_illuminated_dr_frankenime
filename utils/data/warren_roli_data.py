from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from torchvision import transforms
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

    # Use a truetype or opentype font file you have (arial.ttf as an example)
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except IOError:
        print("Default font will be used.")
        font = ImageFont.load_default()

    # Calculate text position to center 'S'
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

# Generate dataset
data, labels = generate_data()
print(f"Generated data shape: {data.shape}, labels shape: {labels.shape}")

# show some examples
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 4, figsize=(10, 10))
for i in range(4):
    axs[i].imshow(data[i], cmap='gray')
    axs[i].set_title(labels[i].item())
plt.show()


def apply_transforms(dataset):
    transformed_data = []
    for sample in dataset:
        transformed_sample = data_transforms(sample)
        transformed_data.append(transformed_sample)
    return torch.stack(transformed_data)

data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=(5, 5)),
    transforms.ToTensor(),
])


data = data.unsqueeze(1)  # Add channel dimension

# Apply data augmentation
transformed_data = apply_transforms(data)

# show some examples
for x in range(10):
    # get next 4 samples
    samples = transformed_data[x*4:(x+1)*4]
    fig, axs = plt.subplots(1, 4, figsize=(10, 10))
    for i in range(4):
        axs[i].imshow(samples[i].squeeze(), cmap='gray')
        axs[i].set_title(labels[i].item())
    plt.show()

# save dataset
dataset = TensorDataset(transformed_data, labels)
torch.save(dataset, "drawing_data.pt")