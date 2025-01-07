import torch
import os
from transformers import DeiTForImageClassificationWithTeacher
import torch.nn as nn
from safetensors.torch import load_file

import torch
import torch.nn as nn
from transformers import DeiTForImageClassificationWithTeacher

# CNN Feature Extractor Class
class CNNFeatureExtractor(nn.Module):
    def __init__(self):  # Fixed constructor name
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # Adjusted for 224x224 input
        self.fc = nn.Linear(128 * 28 * 28, 256)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Define Weighted Fusion Model with DeiT
class Fusion_CNN_DEIT(nn.Module):
    def __init__(self, num_labels):  # Fixed constructor name
        super(Fusion_CNN_DEIT, self).__init__()
        self.cnn = CNNFeatureExtractor()

        # Load the pretrained DeiT model
        self.deit = DeiTForImageClassificationWithTeacher.from_pretrained(
            'facebook/deit-tiny-patch16-224'
        )
        
        # Adjust the classifier for the number of labels
        self.deit.classifier = nn.Linear(self.deit.config.hidden_size, num_labels)
        
        # Learnable weights for CNN and DeiT
        self.weight_cnn = nn.Parameter(torch.tensor(0.5))  # Learnable weight for CNN
        self.weight_deit = nn.Parameter(torch.tensor(0.5))  # Learnable weight for DeiT
        
        # Final classification layer
        self.fc = nn.Linear(256 + self.deit.config.hidden_size, num_labels)  # Adjusted for concatenated features
        self.criterion = nn.CrossEntropyLoss()  # Loss function

    def forward(self, pixel_values, labels=None):
        # Pass inputs through CNN
        cnn_features = self.cnn(pixel_values)

        # Pass inputs through DeiT
        deit_output = self.deit(pixel_values, output_hidden_states=True)
        deit_features = deit_output.hidden_states[-1][:, 0, :]  # CLS token from DeiT

        # Concatenate CNN and DeiT features
        combined_features = torch.cat(
            (self.weight_cnn * cnn_features, self.weight_deit * deit_features), dim=1
        )

        # Classification layer
        logits = self.fc(combined_features)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return {"loss": loss, "logits": logits}



def load_fusion_model(checkpoint_path):
    """
    Function to create the Fusion_CNN_DEIT model and load weights from a checkpoint.
    num_labels is defined as 2 by default.

    Args:
    - checkpoint_path (str): Path to the checkpoint directory containing 'model.safetensors'.

    Returns:
    - model (Fusion_CNN_DEIT): The model with weights loaded from the checkpoint.
    """
    id2label = {0: 'REAL', 1: 'FAKE'}
    label2id = {'REAL': 0, 'FAKE': 1}
    num_labels = 2

    # Instantiate the model
    model = Fusion_CNN_DEIT(num_labels)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    model.deit.config.id2label = id2label
    model.deit.config.label2id = label2id
    print(model.deit.config)
    # Load the model weights from the safetensors file
    model_weights_path = os.path.join(checkpoint_path, "model.safetensors")
    model_weights = load_file(model_weights_path)

    # Load the weights into the model
    model.load_state_dict(model_weights)
    
    return model
import os
import json
import torch
from PIL import Image
import numpy as np


# Define the id2label and label2id mappings
id2label = {0: 'REAL', 1: 'FAKE'}
label2id = {'REAL': 0, 'FAKE': 1}

from PIL import Image

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
from transformers import AutoModelForImageClassification, AutoImageProcessor
from torchvision import transforms
def generate_predictions_json(model, final_directory, output_file='predictions.json'):

    # Define the transformation for image normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match the input size of the model
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=mean, std=std)  # Normalize using the provided mean and std
    ])
    
    # List all image files in the final directory and sort them
    image_files = sorted([f for f in os.listdir(final_directory) if f.endswith(('.jpg', '.png'))])
    # print(f"1, {image_files}")
    
    predictions = []
    
    # Switch model to evaluation mode
    model.eval()
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(final_directory, image_file)
        
        # Open the image
        image = Image.open(image_path).convert('RGB')
        
        # Apply the transformation to the image
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Move the tensor to GPU if available
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
            model = model.cuda()
        
        # Run the image through the model to get outputs
        with torch.no_grad():
            outputs = model(image_tensor)  # Run the image through the model
        
        if 'logits' in outputs:
            logits = outputs['logits']
        else:
            # Handle case where logits are not found as expected
            print("Logits not found in the output, checking raw model output.")
            logits = outputs  # If logits are the entire output, use this
        
        # Apply softmax to get probabilities
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(logits)
        
        # Get the predicted class and its probability
        predicted_class = torch.argmax(probs, dim=1).item()
        predicted_label = id2label[predicted_class]
        
        # Append the result in the required format
        file_number = int(image_file.split('.')[0])
        predictions.append({
            "index": file_number,
            "prediction": predicted_label.lower(),
        })
    predictions = sorted(predictions, key=lambda x: x["index"])

    # Save predictions to JSON file
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=4)
    
    print(f"Predictions saved to {output_file}")

