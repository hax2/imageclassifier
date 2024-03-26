# train.py
import argparse
import json
import os
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict

# Setting up argparse to manage command-line parameters
parser = argparse.ArgumentParser(description='Train a neural network on a set of images.')
parser.add_argument('data_directory', type=str, help='Directory of the data')
parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
parser.add_argument('--arch', type=str, default='vgg16', help='Architecture from torchvision.models to use')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

args = parser.parse_args()

# Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the datasets with ImageFolder
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(args.data_directory, 'train'), transform=train_transforms),
    'valid': datasets.ImageFolder(os.path.join(args.data_directory, 'valid'), transform=valid_test_transforms),
    'test': datasets.ImageFolder(os.path.join(args.data_directory, 'test'), transform=valid_test_transforms)
}

# Using the image datasets, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in ['train', 'valid', 'test']}

# Function to load and return the pre-trained network
def load_pretrained_network(arch):
    model = getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    return model

# Define a new, untrained feed-forward network as a classifier
def build_classifier(input_size, hidden_units):
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('drop', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    return classifier

# Load the pre-trained network
model = load_pretrained_network(args.arch)
model.classifier = build_classifier(model.classifier[0].in_features, args.hidden_units)

# Use GPU if it's available and --gpu was used
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)

# Define the criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Implement a function for the validation pass
def validate_model(model, validloader, criterion):
    model.eval()
    accuracy = 0
    valid_loss = 0
    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        # Calculate accuracy
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return valid_loss, accuracy

# Implement the training loop
for epoch in range(args.epochs):
    model.train()
    running_loss = 0
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    else:
        with torch.no_grad():
            valid_loss, accuracy = validate_model(model, dataloaders['valid'], criterion)
        
        print(f"Epoch {epoch+1}/{args.epochs}.. "
              f"Train loss: {running_loss/len(dataloaders['train']):.3f}.. "
              f"Validation Loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
              f"Validation Accuracy: {accuracy/len(dataloaders['valid']):.3f}")
        model.train()

# Save the checkpoint
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'arch': args.arch,
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict(),
              'classifier': model.classifier,
              'epochs': args.epochs,
              'optimizer_state_dict': optimizer.state_dict()}

torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pth'))
