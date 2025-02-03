# AI Optical Character Recognition

This repository contains the Matlab code that uses 2 custom built machine learning algorithms (K-nearest Neighbour) to perform Optical Character Recognition.
It also includes the EMNIST dataset of 26,000 images and labels on which to train and test the model. A 'summary.pdf' file has been provided to discuss my findings.

## Running the Code -

The code loads the data in from the EMNIST dataset. It splits the data into 50% for training and 50% for testing to avoid overfitting. This means each model runs
13,000 Images so can take several minutes. The models used are -
* Custom K-nearest Neighbour using Euclidean Distance.
* Custom K-nearest Neighbour using Manhattan Distance.
* MatLab K-nearest Neighbour
* MatLab SVM for Multiclass

All that is needed to run the code is the main.m file and the dataset-letters file.

## Files Created -

The code will create a folder called 'Results' in the current directory and create files 'Dataset.png' and 'Confusion.png'.

* 'Dataset.png' - Provides an image of a small sample of EMNIST Images and labels.
  
![ai-dataset](https://github.com/user-attachments/assets/40d403fc-bf10-4dc2-8307-d9b984f8a0f5)

* 'Confusion.png' - Provides confusion charts of the 4 models once they have finished running
  
![ai-results](https://github.com/user-attachments/assets/fce7c891-d795-42d5-a447-c700697ed989)

