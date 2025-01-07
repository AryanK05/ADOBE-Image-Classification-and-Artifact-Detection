# Model Training and Testing Instructions

### Install all the requirements mentioned in `requirements.txt`.

## Downloading the Models
1. Download the desired model files from the provided checkpoints or by training on the train data.
2. Copy the file path of the corresponding main file for the model.

## Training the Models
1. Locate the `clean_dataset` or `clean_dataset_224` file corresponding to the model.
2. Replace the dataset path in the file with the path of the dataset you wish to use for training.
3. Execute the training process by running the following command in the terminal:
   ```bash
   python3 "{path_to_main_file}"
   ```

## Testing the Models
1. Open the main file of the model.
2. Comment out the section of the code related to training.
3. Replace the checkpoint link with the path to the checkpoint of the model you want to test.
4. In the `Test_Data_Pipeline` or `Test_Data_Pipeline_224` file, replace the dataset path with the path of the dataset you wish to use for testing.
5. Execute the testing process by running the following command in the terminal:
   ```bash
   python3 "{path_to_main_file}"
   ```

## Training Specific Models
### DinoV2 with Binary Head, DEiT model from scratch
1. Open the `clean_dataset_224` file.
2. Replace the dataset path in the file with the path to the dataset you wish to train the model on.
3. Execute the training process by running the following command in the terminal:
   ```bash
   python3 "{path_to_model}"
   ```

## Important Note
Ensure that the dataset path is correct before running the command.
