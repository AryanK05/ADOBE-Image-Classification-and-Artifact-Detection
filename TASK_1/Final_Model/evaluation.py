import matplotlib.pyplot as plt
import torch
import time
import numpy as np
from tqdm import tqdm
from datasets import Dataset 
import itertools
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def evaluate_and_plot(outputs, title='Confusion Matrix', figsize=(10, 8)):

    labels_list = ['REAL', 'FAKE']

    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}\n")

    def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(10, 8)):
           
        plt.figure(figsize=figsize)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        fmt = '.0f'
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

    if len(labels_list) <= 150:
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, labels_list, title=title, figsize=figsize)

    print("Classification report:\n")
    print(classification_report(y_true, y_pred, target_names=labels_list, digits=4))


def print_inference_time(trainer, test_data):

    trainer.model.eval()
    trainer.model.to("cpu")

    inference_times = []

    for example in tqdm(test_data, desc="Processing images for inference time"):
 
        image = example['image'].convert("RGB") 
        
        inputs = trainer.processing_class(image, return_tensors="pt").to("cpu")

        start_time = time.time()
        with torch.no_grad():
            _ = trainer.model(**inputs)
        end_time = time.time()

        inference_times.append(end_time - start_time)

    average_time = sum(inference_times) / len(inference_times)
    print(f"Average inference time on CPU for {len(test_data)} images: {average_time:.6f} seconds per image")


    if torch.cuda.is_available():
        device = torch.device("cuda")
        trainer.model.to(device)

        gpu_inference_times = []

        
        for example in tqdm(test_data, desc="Inference on GPU", ncols=100):
            image = example['image'].convert("RGB")  
            
            inputs = trainer.processing_class(image, return_tensors="pt").to(device)
            

            start_time_gpu = time.time()
            with torch.no_grad():
                _ = trainer.model(**inputs)
            end_time_gpu = time.time()

            gpu_inference_times.append(end_time_gpu - start_time_gpu)

        gpu_average_time = sum(gpu_inference_times) / len(gpu_inference_times)
        print(f"Average inference time on GPU for {len(test_data)} images: {gpu_average_time:.6f} seconds per image")
    else:
        print("GPU is not available. Only CPU inference time is measured.")
