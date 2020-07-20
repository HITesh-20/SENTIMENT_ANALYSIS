#SENTIMENT ANALYSIS

#IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score,f1_score

#LOAD DATASET
df=pd.read_csv('C:/Users/Hitesh/Desktop/GITHUB/SENTIMENT ANALYSIS/datasets_39657_61725_amazon_alexa.csv')
print(df)

print(df['variation'].unique())

df1=df['variation'].value_counts()
print("\n\nVARIATIONS\n\n",df1)

#HANDLING MISSING VALUES
df=df.dropna()

#REMOVING LEADING AND ENDING SPACES OF REVIEW
df['verified_reviews']=df['verified_reviews'].apply(lambda x: x.strip())

#TEXT LOWERCASING
df['verified_reviews']=df['verified_reviews'].apply(lambda x: ' '.join(x.lower() for x in x.split()))

#Removing Numbers
df['verified_reviews']=df['verified_reviews'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))

#REMOVED STOP WORDS
nltk.download('stopwords')
stop=stopwords.words('english')
print(len(stop))
df['verified_reviews']=df['verified_reviews'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
print(df['verified_reviews'].head())

#REVIEW LENGTH
df['Review_Length']=df['verified_reviews'].str.len()
print(df.head())
print("\n\n",df.columns)

#POSITIVE AND NEGATIVE FEEDBACKS
positive_feedbacks=df[df['feedback']==1]['verified_reviews']
negative_feedbacks=df[df['feedback']==0]['verified_reviews']

#WORD CLOUD
from wordcloud import WordCloud
plt.figure(figsize=(14,8))
wordcloud1=WordCloud(width=400,height=300, contour_color='black').generate(' '.join(positive_feedbacks))
plt.imshow(wordcloud1)
plt.axis('off')
plt.title('Positive Feedback Reviews',fontsize=40)


plt.figure(figsize=(14,8))
wordcloud1=WordCloud(width=400,height=300, contour_color='black').generate(' '.join(negative_feedbacks))
plt.imshow(wordcloud1)
plt.axis('off')
plt.title('Negative Feedback Reviews',fontsize=40)


#WORD COUNT VECTOR
def word_count(text):
    word_count = {}
    for word in text.split():
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1
    return word_count

df['word_count']=df['verified_reviews'].apply(word_count)
#print("\n\n",df.word_count)

print(df.columns)

#ONE HOT ENCODING
var_dummies=pd.get_dummies(df.variation,prefix='Alexa')
#HERE I'M  TAKING ONLY ONE VARIATIONS ACCORDING TO THE AVERAGE RATING WHICH IS HIGHEST
#YOU CAN TAKE ANY NUMBER OF VARIATIONS BY CHANGING THE NAME OF VARIATIONS IN VAR_DUMMIES
df1=pd.concat([df,var_dummies[['Alexa_Oak Finish ']]],axis='columns')

df1=df1.drop(['variation','date','verified_reviews','word_count'],axis='columns')
print(df1.columns)

#FEATURING
X=df1.drop('feedback',axis='columns')
y=df1.feedback
print(X.shape)
print(y.shape)

#SPLITTING OF DATA INTO TRAIN AND TEST
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#LOGISTIC REGRESSION
lr=LogisticRegression().fit(X_train,y_train)
print("\nACCURACY OF LOGISTIC REGRESSION :",lr.score(X_train,y_train)*100)

#RANDOM FOREST CLASSIFIER
rfc=RandomForestClassifier(n_estimators=100).fit(X_train,y_train)
print("ACCURACY OF RANDOM FOREST CLASSIFIER :",rfc.score(X_train,y_train)*100)

lr_pred=lr.predict(X_test)
rfc_pred=rfc.predict(X_test)

print('Accuracy Score of Logistic Regression',accuracy_score(y_test,lr_pred))
print('Accuracy Score of Random Forest Classifier',accuracy_score(y_test,rfc_pred))

print('F1 Score of Logistic Regression',f1_score(y_test,lr_pred))
print('F1 Score of Random Forest Classifier',f1_score(y_test,rfc_pred))
