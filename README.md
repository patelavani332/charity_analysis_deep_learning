# charity_analysis_deep_learning
Prediction of successful funding using deep learning 
## Overview

The purpose of this projet is to predict whether the applicants will be successful if they are funded by a fictitious company named AlphabetSoup. Here deep learning models are used to predict the outcomes. By training this model on the given dataset of about 34,000 historical donations for both successful and failed outcomes, I used features in dataset to ceate a binary classifier to predict whether applicant would be successful or not if funding is given.  Within this dataset are a number of columns that capture metadata about each organization, such as:

* EIN and NAME—Identification columns
* APPLICATION_TYPE—Alphabet Soup application type
* AFFILIATION—Affiliated sector of industry
* CLASSIFICATION—Government organization classification
* USE_CASE—Use case for funding
* ORGANIZATION—Organization type
* STATUS—Active status
* INCOME_AMT—Income classification
* SPECIAL_CONSIDERATIONS—Special considerations for application
* ASK_AMT—Funding amount requested
* IS_SUCCESSFUL—Was the money used effectively

For this, a neural network is created using Data Manipulation, training and testing sets, and finally the results were evaluated.

## Resources

* Google colboratory
* tensorflow
* scikit-learn
* VS Code

## Results

### Data Processing

Before feeding the data into neural network model, the dependent variable is decided and in this case it is 'IS_SUCCESSFUL' column. Also, 'NAME' and "EIN" columns seemed irrelevant and thus removed. The remaining columns were considered as the features for the model, which will help the neural network model to learn the patterns and make predictions based on the given data.

### Compiling, Training, and Evaluating the Model

Using the tensorflow, a neural network is designed to create a binary classification model which predicts if a funded organization will be successful or not on the basis of features in the dataset. 

Following steps were performed in the Google Colab.

Created a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

Created the first hidden layer.

Added a second hidden layer.

Created an output layer.

Compiled and trained the model.

Evaluated the model using the test data to determine the loss and accuracy.

#### Output
From this neural network model, there was a loss of 56% and accuracy score was 73.03%.
* 2 Hidden Layers
* 80 neurons(Hidden Layer1), 30 neurons(Hidden Layer2)
* Used Relu and Sigmoid Activations Functions since sigmoid is best for binary classifcation problems as this and relu is for nonlinear datasets.

![img1](https://github.com/patelavani332/charity_analysis_deep_learning/assets/120197958/08c401cb-2df2-4b3a-8063-4d23d9d38b71)

### Optimizing the Model

Created a method that creats a new sequential model with hyperparameter options. In this, the kerastuner decides the number of neurons in first layer and also the number of hidden layers and neurons in hidden layers. Then the model is compiled and trained using 20 epochs. On this basis, best hyperparameters model was selected. Using these hyperparameters, a model was created and trained.

#### Output
* From this neural network model, there was a loss of 56% and accuracy score was 73.10%.
* 3 Hidden Layers
* 26 neurons (Layer1), 66 neurons(Hidden Layer1),6 neurons(Hidden Layer1), 36 neurons(Hidden Layer3)
* Used Relu and Sigmoid Activations Functions since sigmoid is best for binary classifcation problems as this and relu is for nonlinear datasets.

![img2](https://github.com/patelavani332/charity_analysis_deep_learning/assets/120197958/b120fa61-a3ea-437a-8bae-69de55ce381a)

## Summary
The accuracy score of my models were 73% and there was a minor improvement in my accuracy score using the optimization technic. Upon careful observations, it is recommended to explore further optimization techniques for the existing model. By fine-tuning parameters, exploring different algorithms, or even exploring more advanced deep learning architectures, we can enhance the model's predictive abilities to fulfill AlphabetSoup's objectives.
