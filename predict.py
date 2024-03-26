# predict.py
import argparse
import json
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from torch import nn
from torchvision import models

# Set up argument parser
parser = argparse.ArgumentParser(description='Predict flower name from an image along with the probability of that name')
parser.add_argument('input', type=str, help='Image path')
parser.add_argument('checkpoint', type=str, help='Model checkpoint path')
parser.add_argument('--top_k', type=int, default=3, help='Return top K most likely classes')
parser.add_argument('--category_names', type=str, help='Use a mapping of categories to real names from a JSON file')
parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

args = parser.parse_args()

# Function to load a checkpoint and rebuild the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=('cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu'))
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.eval()
    return model

# Image processing
def process_image(image_path):
    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    pil_transformed = transform(pil_image)
    return pil_transformed

# Predicting function
def predict(image_path, model, topk=5):
    # Predict the class (or classes) of an image using a trained deep learning model.
    image = process_image(image_path)
    image = image.unsqueeze(0)  # this is for VGG
    with torch.no_grad():
        output = model.forward(image)
    probabilities = torch.exp(output)
    top_probs, top_labs = probabilities.topk(topk)
    top_probs = top_probs.numpy().tolist()[0]
    top_labs = top_labs.numpy().tolist()[0]
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class.get(lab, 'Unknown') for lab in top_labs]
    return top_probs, top_labels

# Main script
if __name__ == '__main__':
    model = load_checkpoint(args.checkpoint)
    probs, classes = predict(args.input, model, args.top_k)
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[str(cls)] for cls in classes]
    print('Predicted Classes:', classes)
    print('Class Probabilities:', probs)
