# data source
# Kaggle: Food choice by BoraPajo
# https://www.kaggle.com/borapajo/food-choices


# import packages
# code citation Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
from tkinter import Grid
import pandas as pd
from pandas.io.stata import value_label_mismatch_doc
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV 

import nltk
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# load in data
survey_data = pd.read_csv('food_coded.csv')
print(survey_data.head(1))

# Explore open end answer
# What is comfort_food
comfort_food_df = survey_data.iloc[:,0:8]
type(comfort_food_df)


stop_words = list(stopwords.words('english'))
#remove stop words
def remove_stop_words(x):
    if len(str(x)) <1:
        return None
    string = " ".join(item.lower() for item in str(x).split() if item.lower() not in stop_words) 
    return string

comfort_food_df['comfort_food'] = comfort_food_df['comfort_food'].apply(lambda x: remove_stop_words(x))


# set up vectorizer
# %%
word_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=800, stop_words='english')
train = comfort_food_df['comfort_food'][0:80].values.tolist()
tfidf = word_vectorizer.fit_transform(train)
tfidf_tokens = word_vectorizer.get_feature_names()
tfidf_df = pd.DataFrame(data=tfidf.toarray(), columns = tfidf_tokens)


# EDA
# Let's create a hypothesis: guessing gender 
# based on breakfast + fav-food + fruit_day + greek food +indian_food + italian_food + thai_food + thai_food  + vitamins

# since all the survey questions are encoded
# Encode the categorical features using One-hot-encoding

# The categorical features
select_data = survey_data[['Gender', 'breakfast', 'fav_food',
                    'fruit_day', 'greek_food', 
                    'indian_food', 'italian_food', 'thai_food',  'vitamins'
]]

categorical_features = ['breakfast', 'fav_food',
                    'fruit_day', 'greek_food', 
                    'indian_food', 'italian_food',
                     'thai_food',  'vitamins'
]

select_data= pd.get_dummies(data=select_data, 
columns=categorical_features)

# Check missing data
missing_values = pd.DataFrame(select_data.isna().sum())
# see if there is any value other than 0 
missing_values[0].unique()


#Since there is no missing value, start splitting training/test/validation data
validation_df = select_data.sample(frac=0.2, random_state=0)
validation_index = validation_df.index

train_test_data = select_data.drop(validation_index)

# reset index
train_test_data = train_test_data.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)

x_valid = validation_df.drop(columns = 'Gender').values
y_valid = validation_df['Gender'].values

x = train_test_data.drop(columns = 'Gender').values
y = train_test_data['Gender'].values

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                test_size=0.2, random_state=0,
                                stratify=y
)

# Create a simple DT classifier
clf = DecisionTreeClassifier(max_depth = 5, 
min_samples_split=2,
max_features='auto')
clf = clf.fit(x_train, y_train)

# predict and compare result
y_pred = clf.predict(x_test)

accuracy_score(y_test, y_pred)

y_valid_pred = clf.predict(x_valid)

accuracy_score(y_valid, y_valid_pred)

# the score does not look very good


# Let's tune the classifier 
parameters = [{
        'classifier__max_depth': [5,7,9],
        'classifier__min_samples_split': [2,3,5],
        'classifier__max_features': ['auto', 'sqrt'],
        'classifier__min_impurity_decrease': [0, 0.01]
}]

pipe = Pipeline([
    ('scaler', StandardScaler()), 
    ('classifier', DecisionTreeClassifier(random_state=0))
])

grid = GridSearchCV(pipe, parameters, cv=2
).fit(x_train, y_train)# data source
# Kaggle: Food choice by BoraPajo

# import packages
from tkinter import Grid
import pandas as pd
from pandas.io.stata import value_label_mismatch_doc
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV 

# load in data
survey_data = pd.read_csv('food_coded.csv')
print(survey_data.head(1))


# EDA
# Let's create a hypothesis: guessing gender 
# based on breakfast + fav-food + fruit_day + greek food +indian_food + italian_food + thai_food + thai_food  + vitamins

# since all the survey questions are encoded
# Encode the categorical features using One-hot-encoding

# The categorical features
select_data = survey_data[['Gender', 'breakfast', 'fav_food',
                    'fruit_day', 'greek_food', 
                    'indian_food', 'italian_food', 'thai_food',  'vitamins'
]]

categorical_features = ['breakfast', 'fav_food',
                    'fruit_day', 'greek_food', 
                    'indian_food', 'italian_food',
                     'thai_food',  'vitamins'
]

select_data= pd.get_dummies(data=select_data, 
columns=categorical_features)

# Check missing data
missing_values = pd.DataFrame(select_data.isna().sum())
# see if there is any value other than 0 
missing_values[0].unique()


#Since there is no missing value, start splitting training/test/validation data
validation_df = select_data.sample(frac=0.2, random_state=0)
validation_index = validation_df.index

train_test_data = select_data.drop(validation_index)

# reset index
train_test_data = train_test_data.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)

x_valid = validation_df.drop(columns = 'Gender').values
y_valid = validation_df['Gender'].values

x = train_test_data.drop(columns = 'Gender').values
y = train_test_data['Gender'].values

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                test_size=0.2, random_state=0,
                                stratify=y
)

# Create a simple DT classifier
clf = DecisionTreeClassifier(max_depth = 5, 
min_samples_split=2,
max_features='auto')
clf = clf.fit(x_train, y_train)

# predict and compare result
y_pred = clf.predict(x_test)

accuracy_score(y_test, y_pred)

y_valid_pred = clf.predict(x_valid)

accuracy_score(y_valid, y_valid_pred)

# the score does not look very good


# Let's tune the classifier 
parameters = [{
        'classifier__max_depth': [5,7,9],
        'classifier__min_samples_split': [2,3,5],
        'classifier__max_features': ['auto', 'sqrt'],
        'classifier__min_impurity_decrease': [0, 0.01]
}]

pipe = Pipeline([
    ('scaler', StandardScaler()), 
    ('classifier', DecisionTreeClassifier(random_state=0))
])

grid = GridSearchCV(pipe, parameters, cv=2
).fit(x_train, y_train)

# predict and compare result
y_pred = grid.predict(x_test)

accuracy_score(y_test, y_pred)

y_valid_pred = grid.predict(x_valid)

accuracy_score(y_valid, y_valid_pred)

# the score did increase but it is still not very good

# Perhaps the hypothesis of preference for different cuisine are not good enough alone to predict gender

# predict and compare result
y_pred = grid.predict(x_test)

accuracy_score(y_test, y_pred)

y_valid_pred = grid.predict(x_valid)

accuracy_score(y_valid, y_valid_pred)

# the score did increase but it is still not very good

# The hypothesis of preference for different cuisine can predict gender is not proved. 
