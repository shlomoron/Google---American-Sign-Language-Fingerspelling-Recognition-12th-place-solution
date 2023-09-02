# Google---American-Sign-Language-Fingerspelling-Recognition-12th-place-solution  
This repository contains my solution (12th place) for the Kaggle competition of [American sign language fingerspelling recognition](https://www.kaggle.com/competitions/asl-fingerspelling) (ASLFR).  

## The steps to reproduce my solution
### 1. Create a TFRecords dataset
TFRecords dataset creation is done in a Kaggle notebook since training on Colab TPU using a Kaggle dataset is best. Also, using Kaggle public datasets is excellent for sharing data across public Colab notebooks, and I will use this method to share the rest of the data in my work, too. [This is the notebook](https://www.kaggle.com/shlomoron/aslfr-parquets-to-tfrecords) for creating the TFRecords dataset, and [here is the dataset](https://www.kaggle.com/datasets/shlomoron/aslfr-tfrecords). To find the path to Kaggle datasets in GCP, see [HERE](https://www.kaggle.com/code/shlomoron/aslfr-gcp-path).
### 2. Train a base model
The first model is trained only on samples with (number of frames with non-nan hands landmarks) > 2*(length of phrase). I do this filtering because there are a lot of corrupted short samples, probably because signers accidentally tap their phone's camera without actually recording themselves. The first model should only be good enough to discriminate the awful samples. I trained a 9,497,577 parameters model for 300 epochs. You can find it at the [ASLFR_base_model.ipynb](https://github.com/shlomoron/Google---American-Sign-Language-Fingerspelling-Recognition-12th-place-solution/blob/main/ASLFR_base_model.ipynb) file.  
The preprocessing involves the normalization of the features by their mean and std. It is useful to save those values for later use instead of calculating them each time. I kept them in a [kaggle dataset](https://www.kaggle.com/datasets/shlomoron/aslfr-means-and-stds).  
### 3. Predict and calculate Levenshtein distances
Use the basic model to predict all the samples and calculate the Levenshtein distances. I deemed samples with very low normalized Levenshtein distance (<0.2) corrupted. This is done [here](https://github.com/shlomoron/Google---American-Sign-Language-Fingerspelling-Recognition-12th-place-solution/blob/main/ASLFR_base_model_predict.ipynb), which is very similar to the base model notebook, except that it loads the weights of the basic model, make predictions and calculate the normalized Levenshtein distance scores instead of training. It loads the model weights from [here](https://www.kaggle.com/datasets/shlomoron/aslfr-base-model), and the computed scores can be found [here](https://www.kaggle.com/datasets/shlomoron/aslfr-base-model-levs). 
### 4. Save the normalized Levenshtein distance scores as TFRecords
I do this in kaggle. The notebook [is here](https://www.kaggle.com/code/shlomoron/aslfr-base-model-levs-tfrecords) and the resulting TFRecords dataset [is here](https://www.kaggle.com/datasets/shlomoron/aslfr-base-model-levs-as-tfrecords).
### 5. Train the final model
The final model notebook is very similat to the base model. The differences are as follow:
1. Increased size: from 9,497,577 parameters to 9,497,577 parameters.
2. Late dropout, 0.8 rate from epoch 15 instead of 0.4 from epoch 0.
3. Batch size increased from 128 to 256.
4. Filtering samplels by the Levenshtein scores calculated at section 3 with a treshild of 0.2 (instead of the Non-nan filtering done in section 1).
5. Maximum frames numbers 340 instead of 380 because with 380 the nodel could not complete the inference in time.
The final model training notebook [is here](https://github.com/shlomoron/Google---American-Sign-Language-Fingerspelling-Recognition-12th-place-solution/blob/main/ASLFR_final_model.ipynb).
### 6. Submit
When submitting I further lowered the maximum frames number to 320 (from 340), since the original model could not complete inference in time (According to prevous experiments it should have, but when I completed the training it couldn't and I had to find a way to make it a bit faster). The post-training max-frame change cause a drop in validation score of ~0.001. The submition nrebook is [HERE](https://www.kaggle.com/code/shlomoron/aslfr-final-model-submission).

