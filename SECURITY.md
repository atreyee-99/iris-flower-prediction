import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App
This app predicts the ***Iris flower*** type!
""")

st.sidebar.header('User Input Parameters') #heading of sidebar
#without sidebar keyword, the header is applied in the main region

def user_input_features(): #custom function to accept all of the four input parameters from the sidebar and it will create a pandas dataframe and the input parameters will be obtained from the sidebar
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4) #format: parameter heading, starting value, ending value and default value
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features() #use the custom function built above to assign into df variable

st.subheader('User Input parameters')
st.write(df) #display the inputs supplied by the user from the sliders(in the sidebar) in tabular form

iris = datasets.load_iris() # loading the iris dataset
X = iris.data #assigning the iris data(comprising the 4 features) into X variable. Will be used later as input argument
Y = iris.target #assigning iris targets into Y variable. They are the class index of number 0, 1, 2

clf = RandomForestClassifier() #creating a classifier variable comprising RandomForest classifier
clf.fit(X, Y) #apply the classifier to build a training model using input arguments X and Y data matrices

prediction = clf.predict(df) #make the prediction
prediction_proba = clf.predict_proba(df) #gives the prediction probability

st.subheader('Class labels and their corresponding index number') 
st.write(iris.target_names) #display the class labels and c orresponding index numbers

st.subheader('Prediction')
st.write(iris.target_names[prediction]) #gives and displays the prediction result
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba) #displays the prediction probability

#open cmd
#type:
#cd desktop
#streamlit rub <filename>
