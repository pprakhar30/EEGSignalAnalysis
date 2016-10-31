# EEG Signal Analysis

This is a part of my final year project where we try to analyse EEG signals [Electroencephlogram][1], which are recording of the electrical activity of the brain from the scalp. 
![Brain Activation Map](https://www.emotiv.com/wp-content/uploads/2016/04/Brain_activity_1.png)

For this experiment we made use of EMOTIV Epoc+ device to collect EEG signals and get the Brain Activation Map videos for individual subjects.
![EMTIV Epoc+](https://i0.wp.com/emotiv.com/wp-content/uploads/2016/01/emotiv_epoc_01.jpg)

## Requirements
- Python 2.7.6
- [Tensorflow][2]
- [scikit-learn][3] : for performance metrics
- [EMOTIV Epoc Brain Activity Map][4]
- [OpenCV2][5]: for Image and Video Processing
- [Pre-Trained VGG16][6]

## Datasets
- The EEG signals and Brain Activation Maps were collected using EMOTIV Epoc+ device mentioned above.
- We collected samples from 13 subjects in which they were shown a list of 25 words and they were supposed to tell whether or not they knew the meaning.The recording of one of the subject was of no use because of too much noise and hence was discarded.
- Following is the distribution of Training and Test Set.
- <b>Training Set</b>: Contains 275 instances of 11 subjects who saw 25 word each
- <b>Test Set</b>: Contains 25 instances of a single subject

## Implementation Details
- In this experiment we used Convolutional Neural Networks(CNNs) and Recurrent Neural Networks(RNNs)
- The model that we used is taken from: [References][7]
- The images are cropped out of BAM videos for each individual which were each 1:46 min long.
- FPS is approximately : 19.09 frames per second.
- there is a 2 second transition period from one word to another.
- We make use of two models for extracting features from CNNs one is `EEG_Model.py` in which we train the CNNs from our images as well
- Other model is `EEG_VGG_Model.py` where we extract the features from `pool5` layer of the popular VGGNet trained on ImageNet
- The Model is trained for 100 epochs for the first model and 50 epochs the second model.

## Performance
###First Model:
- <b>Accuracy</b>: 0.72
- <b>Precision</b>: 0.75
- <b>F1_Score</b>: 0.875
###Second Model:
- <b>Accuracy</b>: 0.28
- One of the major concern over here is that we do not have enough data to train RNNs over a skewed dataset such as ours which is self evident in the second model

## TODO
- Train the model using the images constructed from EEG signals as specified in [References][7].
- We used the BAM videos from the theta frequency bands only we should incorporate Beta and Alpha Frequency bands as well
- Develop Context Based Word Familiarity rather than the Unigram approach that we have made use of, make use of N-gram Word Apporach to understand how an individual percieves the meaning of a word 

## References
- [LEARNING REPRESENTATIONS FROM EEG WITH DEEP RECURRENT-CONVOLUTIONAL NEURAL NETWORKS][7] Paper

## License
MIT


[1]:http://www.medicine.mcgill.ca/physio/vlab/biomed_signals/eeg_n.htm
[2]:https://github.com/tensorflow/tensorflow
[3]:http://scikit-learn.org/stable/index.html
[4]:https://www.emotiv.com/product/epoc-brain-activity-map/
[5]:https://github.com/opencv/opencv
[6]:https://github.com/machrisaa/tensorflow-vgg
[7]:https://arxiv.org/pdf/1511.06448v3.pdf
