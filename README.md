# Google - American Sign Language Fingerspelling Recognition Competition on Kaggle

## Summary

The code in this project was used to develop the models I submitted to Google's American Sign Language Fingerspelling Recognition competition on Kaggle, which ranked in the top 4.71% of submissions, qualifying for a silver placement.

## Project Structure

### Code structure

 - tf_model: module containing code for converting torch model to TFLite
 - torch_model: module containing code for torch model
 - augmentation.py: code for augmenting batches during training
 - dataset.py: code for data processing and dataset creation
 - eda.py: EDA code
 - main.py: trains model on train/val split 
 - plotting.py: code for generating animated gifs from samples
 - show_torch_examples.py: displays labels and predictions for random samples
 - train_on_all.py: trains model on entire dataset
 - utils.py: utility functions
 - validate_tflite.py: checks performance of TFLite model
 - validate_torch.py: checks performance of torch model

### Data Structure / Other

 - conversion: directory used by tf_model module to store intermediate TensorFlow model prior
to tflite conversion
 - plots: directory for storing plots
 - proc_data: processed data
 - raw_data: raw data, downloaded using kaggle API
 - saved_models: stores saved torch models
 - submissions: stores saved TFLite model & inference args for submission
