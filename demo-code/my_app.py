### KICKOFF - CODING AN APP IN STREAMLIT

### import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import warnings
import joblib
from tqdm import tqdm
import time
import ast
import json

# Sklearn ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Text Processing
import string
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Imbalance learning
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Model Evaluation
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

import streamlit as st
import joblib
import shap
shap.initjs()

st.write('Capstone Project Demo - Wells Wang')


#######################################################################################################################################
### LAUNCHING THE APP ON THE LOCAL MACHINE
### 1. Save your *.py file (the file and the dataset should be in the same folder)
### 2. Open git bash (Windows) or Terminal (MAC) and navigate (cd) to the folder containing the *.py and *.csv files
### 3. Execute... streamlit run <name_of_file.py>
### 4. The app will launch in your browser. A 'Rerun' button will appear every time you SAVE an update in the *.py file



#######################################################################################################################################
### Create a title

st.title("H&M Sales Trending Items Prediction")

# Press R in the app to refresh after changing the code and saving here

### You can try each method by uncommenting each of the lines of code in this section in turn and rerunning the app

### You can also use markdown syntax.
#st.write('# Our last morning kick off :sob:')

### To position text and color, you can use html syntax
#st.markdown("<h1 style='text-align: center; color: blue;'>Our last morning kick off</h1>", unsafe_allow_html=True)


#######################################################################################################################################
### DATA LOADING

### A. define function to load data
@st.cache_data # <- add decorators after tried running the load multiple times
def load_data(path):

    df = pd.read_csv(path)

    # Streamlit will only recognize 'latitude' or 'lat', 'longitude' or 'lon', as coordinates

    # df = df.rename(columns={'Start Station Latitude': 'lat', 'Start Station Longitude': 'lon'})     
    # df['Start Time'] = pd.to_datetime(df['Start Time'])      # reset dtype for column
     
    return df

### B. Load first 50K rows
df = load_data("hm_sales_filtered.csv")
prod_df = load_data('prod_df.csv')
X_train_df = load_data('X_train_df.csv')



### C. Display the dataframe in the app
# Custom tokenizer using lemmatizing
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


lemmatizer = WordNetLemmatizer()

ENGLISH_STOP_WORDS = stopwords.words('english')
# Include words we want to clean in stop words
ENGLISH_STOP_WORDS.extend(['none','no', 'nothing','n','a','negative','positive'])

def my_lemma_tokenizer(sentence):
    # # remove punctuation, numbers, and set to lower case
    # sentence = re.sub(r'[^a-zA-Z\s]', '', sentence).lower()

    # remove punctuation and set to lower case
    for punctuation_mark in string.punctuation:
        sentence = sentence.replace(punctuation_mark,'').lower()

    # split sentence into words
    listofwords = sentence.split(' ')
    pos_tags = nltk.pos_tag(listofwords)
    listoflemmatized_words = []

    # remove stopwords and any tokens that are just empty strings
    for word, tag in pos_tags:
        if (not word in ENGLISH_STOP_WORDS) and (word!=''):
            # Lemmatized words

            lemmatized_word = lemmatizer.lemmatize(word,get_wordnet_pos(tag))
            listoflemmatized_words.append(lemmatized_word)

    return listoflemmatized_words


#######################################################################################################################################
### STATION MAP

# st.subheader('Sales Visualization')      

# st.map(df)   


#######################################################################################################################################
### DATA ANALYSIS & VISUALIZATION

### B. Add filter on side bar after initial bar chart constructed

st.sidebar.subheader("User filters")

# Product Type
options = ['Bra', 'Vest top','Trousers']
default_options = ['Bra', 'Trousers']
p_type = st.sidebar.multiselect('Product Type:', options, default_options)

# filter df
df = df[df['product_type_name'].isin(p_type)]
# time series
df['t_dat']=pd.to_datetime(df['t_dat'])
df = df.set_index('t_dat')

# Aggregate by
# User filter for selecting frequency
frequency_options = ['Monthly', 'Weekly', 'Daily']
default_frequency_index = frequency_options.index('Weekly')
selected_frequency = st.sidebar.selectbox('Select Frequency:', frequency_options, index=default_frequency_index)

# Map user selection to corresponding frequency parameter
frequency_mapping = {'Monthly': 'M', 'Weekly': 'W', 'Daily': 'D'}
freq = frequency_mapping[selected_frequency]

# Week slider
# Options
week_options = ['2020-08-30', '2020-08-09', '2020-06-28', '2020-07-05', '2020-07-12',
                '2020-07-19', '2020-07-26', '2020-08-02', '2020-08-16', '2020-08-23',
                '2020-09-06', '2020-09-13', '2020-09-20', '2020-09-27']
# Use select_slider to create a slider with date options
selected_week = st.sidebar.selectbox('Select Week:', options=week_options)






# Comparing sales of different sections
def compare_type_weekly_sales(input_df, types):
    df = input_df.copy()
    df = df.groupby('product_type_name')['units'].resample(freq).sum().reset_index()

    # Create line plot with two categories
    fig = px.line(df, x='t_dat', y='units', color='product_type_name', markers=True, line_dash='product_type_name',
              labels={'t_dat': 'Time', 'units': 'Sales'},
              title=f'{selected_frequency} Sales for Type: ' +' | '.join([f"{i}" for i in types]))
    return fig


fig = compare_type_weekly_sales(df, p_type)

# Display the Plotly plot using st.plotly_chart
st.plotly_chart(fig)


# Top Selling Items
st.subheader('Top Selling Item by Weeks')

def top_by_week(input_df, selected_week):
    df = input_df.copy()
    ref = df.groupby('prod_name')['units'].resample('W').sum().reset_index()
    ref = ref[ref['t_dat']==selected_week]
    top_selling = ref.loc[ref['units'].idxmax(), 'prod_name']
    return top_selling
top_selling = top_by_week(df, selected_week)
st.success(f"Top Selling Item of Week {selected_week.split(' ')[0]}: {top_selling}")


#ML


pipe = joblib.load('trained_pipeline.pkl')
col_trans = joblib.load('col_trans.pkl')

data = prod_df[prod_df['prod_name']==top_selling]
input = col_trans.transform(data)
input = pd.DataFrame(columns = col_trans.get_feature_names_out(), data=input.todense())
st.dataframe(data.reset_index(drop=True).set_index('product_id').T)


st.subheader(f'Features that contribute to predictions: {top_selling}')

prediction = pipe.predict(input)
if prediction == 1:
    st.success('We predict that this is a trending item!')
else:
    st.error('We predict that this is a non-trending item!')


# Fit tree-specific kernel for SHAP
explainer = shap.Explainer(pipe.named_steps['clf'], X_train_df)
# Obtain SHAP values
shap_values = explainer(input)
# Create a waterfall plot using shap.plots.waterfall
shap.plots.waterfall(shap_values[0], max_display=10)
fig, ax = plt.gcf(), plt.gca()

# Display the plot in Streamlit
st.pyplot(fig, clear_figure=True)




### The features we have used here are very basic. Most Python libraries can be imported as in Jupyter Notebook so the possibilities are vast.
#### Visualizations can be rendered using matplotlib, seaborn, plotly etc.
#### Models can be imported using *.pkl files (or similar) so predictions, classifications etc can be done within the app using previously optimized models
#### Automating processes and handling real-time data


#######################################################################################################################################
### MODEL INFERENCE

# st.subheader("Using pretrained models with user input")

# # A. Load the model using joblib
# model = joblib.load('sentiment_pipeline.pkl')

# # B. Set up input field
# text = st.text_input('Enter your review text below', 'bike station was empty')

# # C. Use the model to predict sentiment & write result
# prediction = model.predict({text})





#######################################################################################################################################
### Streamlit Advantages and Disadvantages
