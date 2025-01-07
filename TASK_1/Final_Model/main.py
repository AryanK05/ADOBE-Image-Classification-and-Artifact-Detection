from preprocessing import prepare_image_dataset, apply_transformations  
from evaluation import evaluate_and_plot, print_inference_time  
from model import create_trainer , compute_metrics 
from testing import load_fusion_model , generate_predictions_json

def main():
    
    # datasetpath = ""  # Specify the path to your dataset

    checkpoints = ""  # Specify the path to your model checkpoints

    final_test_folder="" # Specify the path to testing folder which contains images

# All this code is to be uncommented
    # test_size = 0.1  # Proportion of the dataset to be used for testing

    # Prepare the dataset by splitting into train and test sets
    #train_data, test_data = prepare_image_dataset(datasetpath, test_size)

    
    #trainer = create_trainer(train_data, test_data, checkpoints)

    # Make predictions on the test dataset
    #outputs = trainer.predict(test_data)

    # Print the metrics from the predictions
    #metrics = trainer.compute_metrics(outputs)
    #print(metrics)

    # Evaluate and plot the results (confusion matrix, F1 score, etc.)
    #evaluate_and_plot(outputs)

    # Measure and print inference time
    #print_inference_time(trainer,test_data)

    model = load_fusion_model(checkpoints) # This command will load the model weights rom the checkpoints


    print("Model successfully loaded")
    generate_predictions_json(model , final_test_folder)



# Ensure the script runs only if it's executed directly (not when imported)
if __name__ == "__main__":
    main()
