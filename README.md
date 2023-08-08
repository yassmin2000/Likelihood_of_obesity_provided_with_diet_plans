# **Nile University 16th UGRF Special Edition, 2023 Competition**

## **Introduction**
🏆 We proudly stood among the elite "shortlisted teams" in the prestigious Nile University 16th UGRF Special Edition, 2023 Competition.
Our presentation earned its place in the top 10, surpassing more than 121 qualified teams.

🚀 Our project centers on predictive health empowerment. We've harnessed the potential of data to pioneer obesity risk assessment.
By embracing user input encompassing fundamental factors like weight, height, age, and lifestyle-related data, we've crafted an innovative classification model.
Through this innovation, we bestow users with bespoke caloric requirements, personalized meal suggestions, and tailored exercise recommendations.
All this power is conveniently delivered through an intuitive, user-friendly mobile application.

## **Datasets**
🔍 Our obesity risk prediction journey involved harnessing the insights of two primary datasets.
These datasets categorized individuals across a spectrum from "underweight" to "obesity type III."
Integrating these datasets and harmonizing attributes magnified the model's strength, contributing to an enriched user experience.
Moreover, our meal recommendation engine drew from a trio of diverse datasets to cater to individual preferences.
And, we're actively expanding our exercise dataset with inclusive workouts for all users.

## **Data Exploration and Formatting**
📊 We embarked on an in-depth data exploration, unveiling key insights through statistical metrics like mean, variance, and median.
This provided us with a keen understanding of each feature's behavior and its potential impact on our model's prowess.
Adding to this, we engineered novel attributes such as "BMI" and "BMR," distilled from existing data points like weight, height, and gender.
These dynamic attributes fortify the model's resilience and credibility.
The intricate patterns within the data were visually uncovered, guiding our course.

## **Data Cleaning**
🧹 During our data cleaning odyssey, we bid farewell to null values and banished duplications.
Confronting outliers, we summoned the robust "Interquartile Range (IQR) Method."
By exercising this technique, we sculpted a more refined dataset.
This meticulous process underpins our model's accuracy and performance.



## Multiclass Classification Model with PyTorch


### OneHotEncoder Setup:

- The `OneHotEncoder` from the `sklearn.preprocessing` library is used to convert categorical labels into one-hot encoded numerical representations.
- `handle_unknown='ignore'` parameter allows the encoder to handle unknown categories during transformation.
- `sparse_output=False` specifies that the output should be in dense array format.
- The `fit()` method is called on the encoder using the target variable `y` to learn the encoding scheme.
- `ohe.categories_` holds the categories that were encoded.

### One-Hot Encoding:

- The `transform()` method of the encoder is applied to the target variable `y` to transform the categorical labels into one-hot encoded vectors.
- The resulting encoded matrix is stored in `y_hot`.

### Neural Network Model Definition:

- A neural network model for multiclass classification is defined using PyTorch's `nn.Module` as a base class.
- The model consists of three layers:
  - `hidden`: A linear layer with 18 input features and 15 output units.
  - `act`: A ReLU activation function applied after the hidden layer.
  - `output`: Another linear layer with 15 input units and 7 output units (corresponding to the number of classes).

### Model Initialization:

- An instance of the defined `Multiclass` model is created.

### Loss Function and Optimizer Setup:

- The loss function used for training the model is the cross-entropy loss (`nn.CrossEntropyLoss()`), suitable for multiclass classification.
- The Adam optimizer (`optim.Adam()`) is used to update the model parameters during training, with a learning rate of 0.001.

### Data Preparation:

- The pandas DataFrame `X` and the one-hot encoded `y_hot` are converted into PyTorch tensors with the appropriate data types (`torch.tensor()`).
- The training and testing data are split using `train_test_split()` from an assumed import (note: `train_test_split` seems to be missing from the code).


### Training and Evaluation Loop Explanation

This code snippet demonstrates the process of training and evaluating a neural network model for a multiclass classification task using PyTorch. The process is outlined as follows:

#### Training Loop:

- A certain number of training epochs (`n_epochs`) is defined, and the batch size (`batch_size`) for training is set.

- Metrics such as `best_acc`, `best_weights`, `train_loss_hist`, `train_acc_hist`, `test_loss_hist`, and `test_acc_hist` are initialized for tracking performance.

- The training loop iterates through each epoch. Within each epoch, the training data is divided into batches (`batches_per_epoch`).

- The model is set to training mode (`model.train()`), and for each batch, the following steps are performed:
  - Extract a batch of input data (`X_batch`) and corresponding target labels (`y_batch`) from the training dataset.
  - Convert the one-hot encoded target labels (`y_batch`) to class indices.
  - Perform a forward pass through the model (`model(X_batch)`) to obtain predicted class probabilities (`y_pred`).
  - Calculate the loss using the cross-entropy loss function (`loss_fn`) with the predicted probabilities and target class indices.
  - Perform backward pass, compute gradients, and update the model weights using the optimizer (`optimizer.step()`).
  - Compute and store training metrics such as loss and accuracy for the current batch.

- After each epoch's batch iteration, the model is switched to evaluation mode (`model.eval()`).

#### Evaluation Loop:

- The model's predictions are obtained using the test data (`X_test`).

- Similar to the training loop, the one-hot encoded target labels (`y_test`) are converted to class indices.

- Cross-entropy loss (`ce`) and accuracy (`acc`) are calculated based on the model's predictions and the ground truth class indices.

- The computed metrics are stored in the `test_loss_hist` and `test_acc_hist` lists.

#### Best Model Tracking:

- The model's state dictionary is copied (`copy.deepcopy(model.state_dict())`) when the accuracy in the current epoch surpasses the `best_acc` obtained so far.

- The model's state dictionary with the best weights is loaded back into the model using `model.load_state_dict(best_weights)`.

![model accuracy](https://i.ibb.co/7JTn96c/accuracy-graph.png)

## **Creating and Evaluating Decision Tree and Random Forest Models**

In this code snippet, we walk through the process of building and assessing Decision Tree and Random Forest models for classification tasks.

### **Decision Tree Model**

- We create an instance of the `DecisionTreeClassifier()` as `model_dt`.

- The model is trained using the training data (`X_train` and `y_train`) using the `.fit()` method.

- Predictions are made on the training data using the trained Decision Tree model, and the results are stored in `y_pred_train_dt`.

- Accuracy and confusion matrix for the training set are calculated and stored in `accuracy_train_dt` and `confusion_matrix_train_dt`, respectively.

- Similar steps are repeated for predictions and metrics evaluation on the test set (`X_test` and `y_test`).

### **Random Forest Model**

- We create an instance of the `RandomForestClassifier()` as `model_rf`.

- The model is trained using the training data (`X_train` and `y_train`) using the `.fit()` method.

- Predictions are made on the training data using the trained Random Forest model, and the results are stored in `y_pred_train_rf`.

- Accuracy and confusion matrix for the training set are calculated and stored in `accuracy_train_rf` and `confusion_matrix_train_rf`, respectively.

- Similar steps are repeated for predictions and metrics evaluation on the test set (`X_test` and `y_test`).

## **Accuracy Results**

### **Decision Tree Model**

#### **Training Set:**
- Accuracy: 100%
#### **Test Set:**
- Accuracy: 99.37%

### **Random Forest Model**

#### **Training Set:**
- Accuracy: 100%
#### **Test Set:**
- Accuracy: 99.62%

 The consistently high accuracy scores and well-distributed confusion matrices underscore the models' robust performance across training and test data.

## **Cross-Validation and Model Evaluation**

In this section, we delve into the application of k-fold cross-validation to assess the performance of the trained Decision Tree and Random Forest models.

### **Decision Tree Model**

We utilize the `cross_val_score()` function from the `sklearn.model_selection` module to execute 5-fold cross-validation on the Decision Tree model (`model_dt`).
- The training data (`X_train` and `y_train`) is used for cross-validation.
- The resulting cross-validation scores are stored in `cv_scores_dt`.

We then print the individual cross-validation scores and calculate the average accuracy of the Decision Tree model using `.mean()` on `cv_scores_dt`.

### **Results:**
- Mean accuracy for Decision Tree: 0.998911562611065

### **Random Forest Model**

Next, we create an instance of the `RandomForestClassifier()` as `model_rf`.

We similarly apply 5-fold cross-validation using the `cross_val_score()` function on the Random Forest model.
- The training data (`X_train` and `y_train`) is used for cross-validation.
- The resulting cross-validation scores are stored in `cv_scores_rf`.

We print the individual cross-validation scores and calculate the average accuracy of the Random Forest model using `.mean()` on `cv_scores_rf`.

### **Results:**
- Mean accuracy for Random Forest: 0.9989100817438692

## **Feature Importance**

Feature importance is a crucial concept in machine learning that helps us understand the significance of different input features in making predictions. In the context of our obesity risk prediction project, feature importance sheds light on which attributes play a substantial role in determining an individual's obesity type.

### **Why is Feature Importance Important?**

Feature importance allows us to identify and prioritize the most influential factors that contribute to our model's predictions. This understanding is valuable for several reasons:
- **Insights into Relationships:** Feature importance can reveal which attributes are closely related to the target variable (obesity type). These insights help us better understand the factors that contribute to different obesity categories.

- **Model Interpretability:** By highlighting significant features, we can offer users a more interpretable explanation of how their data influences the model's predictions. This enhances the transparency and trustworthiness of our prediction system.

- **Feature Selection:** Feature importance guides us in making informed decisions about feature selection. If certain attributes are found to have little impact on predictions, we might consider excluding them to streamline our model and reduce complexity.

![feature importance](https://i.ibb.co/tLVdzDy/download.png)

## Model Prediction Functions



