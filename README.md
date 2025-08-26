# Hearing-Well-Being-Survey-Analysis-with-Machine-Learning

Objective:

The goal of this project is to analyze a survey dataset related to hearing well-being and build machine learning models to predict whether a person is interested in a hearing app (Interest_in_Hearing_App).



Step 1: Data Preprocessing

Loading data: The dataset is loaded into a DataFrame called df.

Handling missing values:

Rows or columns with all missing values are dropped (df.dropna(axis=1, how="all")).

Missing values are then filled using a SimpleImputer with the most frequent value strategy. This replaces missing data points with the most common value in the respective columns.

The goal is to ensure no missing values remain for the model training stage.



Step 2: Define Features and Target

The target variable for prediction is "Interest_in_Hearing_App".

Features (X) are all other columns excluding the target.

Target (y) is the "Interest_in_Hearing_App" column.



Step 3: Train-Test Split

The dataset is split into training and testing sets using:


20% of the data is held out for testing the model after training.

Random state ensures reproducibility.



Step 4: Define Models

You are training six classification models:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Naive Bayes

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)



Step 5: Model Training and Evaluation

Each model is trained (model.fit(X_train, y_train)) on the training data.

Predictions are made on the test data (model.predict(X_test)).

Accuracy is calculated using accuracy_score.

Accuracies are stored in a dictionary.



Step 6: Issue Observed

Logistic Regression raises a Convergence Warning indicating the model didn’t converge within 1000 iterations.

This usually means:

Data may need scaling (especially important for Logistic Regression, SVM).

Model parameters like max_iter may need to be increased.

The warning suggests scaling data using preprocessing techniques for better convergence.



Step 7: Results

Accuracies of models are quite low overall, with Logistic Regression and Naive Bayes performing the best (~46-47% accuracy).

The other models perform worse (between 34% to 40% accuracy).

This indicates that the models are not very effective with the current features or data.



Step 8: Visualization

A bar chart is plotted showing the accuracy of each model.

Logistic Regression and Naive Bayes stand out slightly above the others but still under 50% accuracy.



Summary and Next Steps:

The project has successfully implemented a pipeline to preprocess data, split it, train multiple models, and evaluate their performance.

However, the model performance is low, suggesting improvements are needed.

Recommended improvements:

Scale the features using StandardScaler or MinMaxScaler.

Perform feature selection or engineering to find better predictors.

Tune hyperparameters of models for better performance.

Try other more advanced models like Gradient Boosting or XGBoost.

Use cross-validation for more robust evaluation.


