![NU certificate](WhatsApp-Image-2024-06-08-at-02.35.35_761b5f15.jpg)


# **Nile University 16th UGRF Special Edition, 2023 Competition**

## **Introduction**
🏆 We proudly stood among the elite "shortlisted teams" in the prestigious Nile University 16th UGRF Special Edition, 2023 Competition.
Our presentation earned its place in the top 10, surpassing more than 121 qualified teams.

🚀  In essence, our project revolves around enabling users to input their information, which comprises fundamental details like weight, height, age, and lifestyle-related data. Leveraging our developed classification model, we provide the user with personalized calorie requirements for maintenance, along with suggestions for recommended meals and exercises. All of this is made accessible through a user-friendly mobile application.

## **Datasets**
🔍 For predicting obesity types, we utilized two primary datasets that can be classified into various categories such as "underweight, normal weight, overweight types I & II, obesity types I & II & III." To enhance the robustness of our model and provide the user with more comprehensive insights, we combined these datasets using shared, appropriate columns. This integration not only fortified the model's performance but also introduced additional features to enrich the user experience.

Incorporating diversity into our meal recommendations was achieved by integrating three distinct datasets. This approach aimed to introduce a range of options and cater to individual user preferences effectively. As for the exercise dataset, there is an ongoing effort to expand its content by including a broader spectrum of workouts that are accessible to all users.

## **Data Exploration and Formatting**
📊 We conducted a comprehensive analysis of the data, calculating key statistical metrics like mean, variance, and median. This process provided insights into the behavior of each feature and its potential impact on the model. To further enhance the model's accuracy and reliability, we introduced additional features such as "BMI" and "BMR." These new attributes were derived from existing data points like weight, height, and gender. The inclusion of these derived features aimed to bolster the model's robustness and validity. Additionally, we visualized the data distribution to gain a deeper understanding of its patterns and trends.

We also harmonized common attributes across the two datasets, ensuring uniform labels and attributes. This standardization was a crucial step in facilitating the seamless merging of the datasets, enabling a cohesive and coherent integration process.

We converted categorical data into numerical form by leveraging the "fit_transform" method provided by the sklearn preprocessing library. This transformation allowed us to represent categorical information in a format suitable for analytical processes and model training.

## **Data Cleaning**
🧹 During the data cleaning phase, we systematically eliminated both null values and duplicate entries. To address outliers that were detected, we applied the interquartile range (IQR) method. This method involves removing data points lying beyond a certain range, contributing to a more refined dataset that is better suited for analysis and subsequently enhances the accuracy of our model.



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

## App Design
- **Login & Sign-up pages:**
![feature importance](https://i.ibb.co/yYVrkMz/Picture1.png)
- **User data:**
![feature importance](https://i.ibb.co/xMtjKxG/Picture2.png)
- **Calories & Recommended meals and exercises:**
![feature importance](https://i.ibb.co/j5Nyd1M/Picture3.png)

## 🙅🏻‍♂️Contributors <a name = "Contributors"></a>
<table>
  <tr>
    <td align="center">
    <a href="https://github.com/MohamedMandour10" target="_black">
    <img src="https://avatars.githubusercontent.com/u/115044826?v=4" width="150px;" alt="Mohamed Elsayed Eid"/>
    <br />
    <sub><b>Mohamed Elsayed Eid</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/melsayed8450" target="_black">
    <img src="https://avatars.githubusercontent.com/u/100236901?v=4" width="150px;" alt="Mohamed Elsayed Ali"/>
    <br />
    <sub><b>Mohamed Elsayed Ali</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/MahmoudMagdy404" target="_black">
    <img src="https://avatars.githubusercontent.com/u/83336074?v=4" width="150px;" alt="Mahmoud Magdy"/>
    <br />
    <sub><b>Mahmoud Magdy</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/yassmin2000" target="_black">
    <img src="https://avatars.githubusercontent.com/u/105241119?v=4" width="150px;" alt="Mahmoud Magdy"/>
    <br />
    <sub><b>Yassmin Sayed</b></sub></a>
    </td>
      <td align="center">
    <a href="https://github.com/SaraElsaggan" target="_black">
    <img src="https://avatars.githubusercontent.com/u/104657535?v=4" width="150px;" alt="Mahmoud Magdy"/>
    <br />
    <sub><b>Sara Elsaggan</b></sub></a>
    </td>
      </tr>
 </table>



