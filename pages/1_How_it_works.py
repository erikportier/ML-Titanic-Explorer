import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn import set_config
from joblib import dump, load
from sklearn.tree import export_graphviz
from subprocess import call




def transform_cols():
    tf1 = ColumnTransformer(transformers=[("SexEncoder", OrdinalEncoder(), ['sex']),
                    ("OneHotEncoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ['embarked']),
                ("Scale", MinMaxScaler(),['age', 'fare'])], remainder = "passthrough")
    return tf1

def train_model(preprocessing_pipeline, model_class,filename):
            model = make_pipeline(preprocessing_pipeline,model_class)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc_train = model.score(X_train, y_train)
            acc_test = accuracy_score(y_test, y_pred)
            dump(model, filename +'.joblib') 
            
            return y_pred, acc_train, acc_test

titanic_df = pd.read_excel("./data/titanic_dataset.xls")
X = titanic_df.drop(['survived','name','ticket'], axis = 1)
y = titanic_df['survived']



    # create model input
#if st.button('Explore Data'):
st.write("### First 5 rows of data: 📊")
st.write(titanic_df.head())

if st.button('Remove Missing Values 🧩'):
    drop_na = titanic_df.dropna()
    df = pd.concat([titanic_df.isna().sum(), drop_na.isna().sum()], axis =1)
    df.columns = ["Before" ,"After"]
    st.write(df)



split_val = st.slider("What percentage of the data will be used for testing?", min_value=.10, max_value=.50)

if st.button('Split Data: ✂️'):
    titanic_df = titanic_df.dropna()
    X = titanic_df.drop(['survived','name','ticket'], axis = 1)
    y = titanic_df['survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= split_val, stratify = y, random_state= 42)
    st.write('X_train size: ' + str(X_train.shape) +' , ' + 'y_train size: ' +str(y_train.shape))
    st.write('X_test size: ' + str(X_test.shape) +' , ' + 'y_test size: ' +str(y_test.shape))

if st.button('Tranform Data 🪄'):
    set_config(transform_output="pandas")
    titanic_df = titanic_df.dropna()
    X = titanic_df.drop(['survived','name','ticket'], axis = 1)
    y = titanic_df['survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= split_val, stratify = y, random_state= 42)
    tf = transform_cols()
    df = tf.fit_transform(X_train)
    st.write(df)

option = st.selectbox( 'Which model woul you like to try?',
('Logistic Regression', 'Decision Tree', 'Random Forest')) 
if st.button('Train & Evaluate Model'):
    titanic_df = titanic_df.dropna()
    X = titanic_df.drop(['survived','name','ticket'], axis = 1)
    y = titanic_df['survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= split_val, stratify = y, random_state= 42)
    tf = transform_cols()

    if option == "Logistic Regression":
        y_pred, acc_train, acc_test = train_model(tf, LogisticRegression(),option)
        st.write("**Model used:** " + str(option))
        st.write("**Accuracy (Training set):** "  + str(acc_train))
        st.write("**Accuracy (Test set):** "  + str(acc_test))
    elif option == "Decision Tree":  
        y_pred, acc_train, acc_test = train_model(tf, DecisionTreeClassifier(),option)
        st.write("**Model used:** " + str(option))
        st.write("**Accuracy (Training set):** "  + str(acc_train))
        st.write("**Accuracy (Test set):** "  + str(acc_test))
    elif option == "Random Forest":
        y_pred, acc_train, acc_test = train_model(tf, RandomForestClassifier(),option)
        st.write("**Model used:** " + str(option))
        st.write("**Accuracy (Training set):** "  + str(acc_train))
        st.write("**Accuracy (Test set):** "  + str(acc_test))
    else:
        st.write("Please pick a model!")

    fig, ax  = plt.subplots()
    cm = confusion_matrix(y_pred, y_test)
    ax = sns.heatmap(cm, annot=True, fmt='d',xticklabels = ['Survived', 'Did Not Survive'], yticklabels = ['Survived', 'Did Not Survive'])
    st.pyplot(fig)

