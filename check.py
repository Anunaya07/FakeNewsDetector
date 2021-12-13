import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import PassiveAggressiveClassifier
import joblib

# nltk.download()
porter = SnowballStemmer('english')
myTfidf = TfidfVectorizer(max_df=0.7)

def stemIt(text):
    return [porter.stem(word) for word in text]

def stopIt(text):
    return [word for word in text if len(word) > 2]


data = pd.read_csv("True.csv")
# data = data.head(10)
data['target']=1
data['text'] = data['text'].apply(word_tokenize)
data['text'] = data['text'].apply(stemIt)
data['text'] = data['text'].apply(stopIt)
data['text'] = data['text'].apply(' '.join)


X_train,X_test,y_train,y_test = train_test_split(data['text'],data['target'],test_size=0.25)
print(X_test)
tfidf_test = myTfidf.transform(X_test)
print(tfidf_test[0])






