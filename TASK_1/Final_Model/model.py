import os
import torch
from transformers import Trainer, AutoImageProcessor, TrainingArguments, AutoModelForImageClassification, AutoProcessor, TrainerState , DeiTForImageClassificationWithTeacher
import evaluate  
from safetensors.torch import load_file
import torch.nn as nn
accuracy = evaluate.load("accuracy")

# CNN Feauture Extractor ( better at extraction of local features )
class CNNFeatureExtractor(nn.Module):
    def __init__(self):  
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

# Combined Model 
# Transformers are better for global dependencies
class Fusion_CNN_DEIT(nn.Module):
    def __init__(self, num_labels): 
        super(Fusion_CNN_DEIT, self).__init__()
        self.cnn = CNNFeatureExtractor()

        # Load the pretrained DeiT model
        self.deit = DeiTForImageClassificationWithTeacher.from_pretrained(
            'facebook/deit-tiny-patch16-224'
        )

       
        self.deit.classifier = nn.Linear(self.deit.config.hidden_size, num_labels)

        # Learnable weights for CNN and DeiT
        self.weight_cnn = nn.Parameter(torch.tensor(0.5)) 
        self.weight_deit = nn.Parameter(torch.tensor(0.5))  

        # Final classification layer
        self.fc = nn.Linear(256 + self.deit.config.hidden_size, num_labels) 
        self.criterion = nn.CrossEntropyLoss() 

    def forward(self, pixel_values, labels=None):
        
        cnn_features = self.cnn(pixel_values)

       
        deit_output = self.deit(pixel_values, output_hidden_states=True)
        deit_features = deit_output.hidden_states[-1][:, 0, :] 

        
        combined_features = torch.cat(
            (self.weight_cnn * cnn_features, self.weight_deit * deit_features), dim=1
        )

        
        logits = self.fc(combined_features)

        
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return {"loss": loss, "logits": logits}



def create_trainer(train_dataset, eval_dataset, checkpoint_path="facebook/deit-tiny-distilled-patch16-224"):

    #  Define labels and their mappings directly
    id2label = {0: 'REAL', 1: 'FAKE'}
    label2id = {'REAL': 0, 'FAKE': 1}

    # Initialize the Weighted Fusion Model
    num_labels = 2  # Here , we have only two labels
    model = Fusion_CNN_DEIT(num_labels)

    # Count trainable parameters
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    model.deit.config.id2label = id2label
    model.deit.config.label2id = label2id
    print(model.deit.config)

    processor = AutoImageProcessor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')



    # Define the name of the model, which will be used to create a directory for saving model checkpoints and outputs.
    model_name = "FUSION_MODEL"

    # Define the number of training epochs for the model.
    num_train_epochs = 2
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Creating an instance of TrainingArguments to configure training settings.
    args = TrainingArguments(
        
        output_dir=os.path.join(current_dir, model_name),

        logging_dir=os.path.join(current_dir, 'logs'),

        evaluation_strategy="epoch",

        learning_rate=1e-6,

        per_device_train_batch_size=200,

        per_device_eval_batch_size=400,

        num_train_epochs=num_train_epochs,

        weight_decay=0.02,

        warmup_steps=50,

        remove_unused_columns=False,

        save_strategy='epoch',

        load_best_model_at_end=True,

        save_total_limit=1,

        report_to="none" 
    )
        

    

    # Loading weights from the safetensor file
    model_weights = load_file( os.path.join(checkpoint_path, "model.safetensors"))


    # Loading the weights into the model
    model.load_state_dict(model_weights)

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    return trainer


def compute_metrics(eval_pred):
        predictions = eval_pred.predictions.argmax(axis=1)
        label_ids = eval_pred.label_ids
        acc_score = accuracy.compute(predictions=predictions, references=label_ids)['accuracy']
        return {"accuracy": acc_score}

def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example['label'] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

