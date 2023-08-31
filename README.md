# Google---American-Sign-Language-Fingerspelling-Recognition-12th-place-solution  
This is my solution (12th place) for the kaggle competition of [American sign language fingerspelling recognition](https://www.kaggle.com/competitions/asl-fingerspelling) (ASLFR).  

## The steps to reproduce my solution
### 1. Create a TFRecords dataset
This is done in a kaggle notebook, since training on Colab TPU is best done by using a kaggle dataset. [This is the notebook](https://www.kaggle.com/shlomoron/aslfr-parquets-to-tfrecords) and [here is the dataset](https://www.kaggle.com/datasets/shlomoron/aslfr-tfrecords).  
### 2. Train a base model
The first model is trained only on samples with (number of frames with non-nan hands landmarks) > 2*(length of phrase). This is done because there are a lot of corrupted short samples, probably the result of signers accidentaly tapping their phone's camera without actually recording themselves. The first model should only be good enough to discriminate the extremely bad samples. I trained a 9,497,577 parameters model for 300 epoch. You can find it at the [ASLFR_base_model.ipynb](https://github.com/shlomoron/Google---American-Sign-Language-Fingerspelling-Recognition-12th-place-solution/blob/main/ASLFR_base_model.ipynb) file.  
The preprocessing involves normalization of the features by their mean and std. It is useful to save those values for later use instead of calculating it each time. I saved them in a [kaggle dataset](https://www.kaggle.com/datasets/shlomoron/aslfr-means-and-stds).  
### 3. Predict and calculate Levenshtein distances
Use the basic model to make predictions for all the samples and calculate the Levenshtein distance. Samples with very low normalized Levenshtein distance (0.2) would be deemed as corrupted samples. This is done [here](https://github.com/shlomoron/Google---American-Sign-Language-Fingerspelling-Recognition-12th-place-solution/blob/main/ASLFR_base_model_predict.ipynb), which is the very similar to the base model notebook, except that it loads the weights of the model and make predictions and calculate the normalized Levenshtein distance scores instead of training. It loads the model weights from [here](https://www.kaggle.com/datasets/shlomoron/aslfr-base-model) and the calculated scores can be found [here](https://www.kaggle.com/datasets/shlomoron/aslfr-base-model-levs).  
### 4. Save the normalized Levenshtein distances scores as TFRecords
This is done in kaggle. The notebook [is here](https://www.kaggle.com/code/shlomoron/aslfr-base-model-levs-tfrecords) and the resulting TFRecords dataset [is here](https://www.kaggle.com/datasets/shlomoron/aslfr-base-model-levs-as-tfrecords).
### 5. Train the final model
The final model notebook is very similat to the base model. The differences are as follow:
1. Increased size: from 9,497,577 parameters to 9,497,577 parameters.
2. Late dropout, 0.8 rate from epoch 15 instead of 0.4 from epoch 0.
3. Batch size increased from 128 to 256.
4. Filtering samplels by the Levenshtein scores calculated at section 3 with a treshild of 0.2 (instead of the Non-nan filtering done in section 1).
The final model training notebook [is here](https://github.com/shlomoron/Google---American-Sign-Language-Fingerspelling-Recognition-12th-place-solution/blob/main/ASLFR_final_model.ipynb). 
