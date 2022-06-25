import numpy as np
import pandas as pd
from pyparsing import col
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from PIL import Image
setosa_image = Image.open("setosa.jpg")
versicolor_image = Image.open("versicolor.jpg")
virginica_image = Image.open("virginica.jpg")

#st.image(image, caption='Sunrise by the mountains')

iris = load_iris()
target_names = iris.target_names
X=iris.data
y=iris.target
st.write(target_names)
clf=DecisionTreeClassifier()
clf.fit(X,y)

sepal_legth = st.sidebar.number_input("Sepal length",4.3,7.9,5.4)
sepal_width = st.sidebar.number_input("Sepal width",2.0,4.4,3.4)
petal_legth = st.sidebar.number_input("Petal length",1.0,6.9,1.3)
petal_width = st.sidebar.number_input("Petal width",0.1,2.5,0.2)

data = dict(
    sepal_legth = sepal_legth,
    sepal_width = sepal_width,
    petal_legth = petal_legth,
    petal_width = petal_width
)

df = pd.DataFrame(data, index = ["Length in mm"])


st.title('Prediction of iris flower type based on user input')



st.text("Thank you for providing your input")
st.dataframe(df.T)



st.header("This part shows predicted type of flower")
st.text('Here is your prediction:')

prediction=clf.predict(df)
predict_prob=clf.predict_proba(df)
max_prob = np.max(predict_prob) * 100

st.markdown(f"The flower you are trying to predict is **{target_names[prediction][0]}** with probability of **{max_prob}** %")

image_dict= {'setosa': setosa_image,'versicolor':versicolor_image,'virginica':virginica_image}

st.image(image_dict[target_names[prediction][0]], caption=(f"The flower you are trying to predict is {target_names[prediction][0]} with probability of {max_prob} %"))
