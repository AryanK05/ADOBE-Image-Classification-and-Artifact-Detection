
# Instructions for Training and Testing the Model

## Prerequisites
First, ensure you have the required libraries installed. Run the following command to import all necessary libraries:

```bash
pip install -r requirements.txt
```

## Training the Model

1. **Directory Structure for Training**:
   - Ensure the dataset directory has only two folders: `FAKE` and `REAL`.
   - These folders should contain the respective images for training.

2. **Extract Checkpoints**:
   - Extract the provided checkpoints into the current folder.

3. **Load Pretrained Weights**:
   - Update the paths to the checkpoints and dataset in the `main.py` file. The checkpoints of the pre-trained model has been uploaded in the folder.
   - Use the path to the saved checkpoint to load the pretrained weights for further training.

---

## Testing the Model

1. **Directory Structure for Testing**:
   - Modify the `main.py` file to include the correct path to the test folder.
   - Ensure this folder contains only the images to be tested.

2. **Checkpoint Path**:
   - Provide the path to the saved checkpoint file to use the pretrained weights for testing.

---

### Notes
- Uncomment the necessary commands in code to initiate training or testing as needed.
- Ensure all folder paths and file structures are correctly set up before running the model.
