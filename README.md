# Google---American-Sign-Language-Fingerspelling-Recognition-12th-place-solution  
This is my solution (12th place) for the kaggle competition of [American sign language fingerspelling recognition](https://www.kaggle.com/competitions/asl-fingerspelling) (ASLFR).  

## The steps to reproduce my solution
### 1. Create TFRecords dataset
This is done in a kaggle notebook, since training on Colab TPU is best done by using a kaggle dataset. [This is the notebook](https://www.kaggle.com/shlomoron/aslfr-parquets-to-tfrecords) and [here is the dataset](https://www.kaggle.com/datasets/shlomoron/aslfr-tfrecords).  
### 2. Train a base model
The first model is trained only on the samples with (number of frames with non-nan hands landmarks) > 2*(length of phrase). This is done because there are a lot of corrupted short samples, probably the result of signers accidentaly tapping their phone's camera without actually recording themselves.
