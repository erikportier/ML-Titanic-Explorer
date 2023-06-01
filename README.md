
# ML Titanic Explorer

![iceberg_image](https://plus.unsplash.com/premium_photo-1676573201187-fb4546c5bf8a?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1032&q=80)

The goal of this app is to walk-through some of the main steps in creating an machine learning pipeline using as a base the Titanic dataset in a no-code way.

## The App

You can dive right into the app which is deployed on Streamlit.io by following this link: https://vriveraq-ml-titanic-explorer-hello-ix9qg8.streamlit.app/

## Data

We train our models on a subset of the Titanic dataset from Kaggle. Which includes the following features:

* pclass	- passenger class
* survived	- if the passenger survived 1, else if the passenger did not survive 0
* name	- name of the passenger
* sex	- sex of the passenger (male or female)
* age	- age of the passenger (0 to 100)
* ticket	- ticket number
* fare	- price of ticket
* embarked - port which the passenger embarked on

## Missing Data and Transformations

For this example, we choose to drop the name and ticket column as they did not provide useful information for training the model. In addition, all missing values (263 in the age column, 2 in the fare column, 1 in the embarked column) were removed.

To transform our catergorical columns into numerical columns we used an `OrdinalEncoder()` for the sex and a `OneHotEncoder()` for the embarked column.  Finally, we used a `MinMaxScaler()` to transform the fare to values between 0 and 1 (which is particularly important for the Logistic Regression model used in the next section).

## Models & Evaluation

We focused on testing 3 models for this classification task, due to their interpretability, mainly: `Logistic Regression`, `Decision Tree`, and `RandomForest`. We evaluate each model on the accuracy score and using the confusion matrix. 



