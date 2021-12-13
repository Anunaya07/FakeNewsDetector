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

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
fake['target']=0
true['target']=1

data = pd.concat([fake,true],axis=0)
data = data.reset_index(drop=True)
# print(data.head())s
data = data.drop(['title','subject','date'], axis=1)
# print(data.columns)

data['text'] = data['text'].apply(word_tokenize)
data['text'] = data['text'].apply(stemIt)
data['text'] = data['text'].apply(stopIt)
data['text'] = data['text'].apply(' '.join)


X_train,X_test,y_train,y_test = train_test_split(data['text'],data['target'],test_size=0.25)

# print(x_train.head())
# print("\n")
# print(y_train)

tfidf_train = myTfidf.fit_transform(X_train)
tfidf_test = myTfidf.transform(X_test)

#Logistic Regression
model_1 = LogisticRegression(max_iter=900)
model_1.fit(tfidf_train, y_train)
pred_1 = model_1.predict(tfidf_test)
print("Logistic Regression predicts:\n")
print(pred_1)
cr1 = accuracy_score(y_test, pred_1)
print("Logistic Regression accuracy: ", cr1*100)
joblib.dump(model_1,'fakeNewsDetector1.pkl')

# Passive Aggressive Classifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)
y_pred = model.predict(tfidf_test)
print("\nPassive Aggressive Classifier:\n")
print(y_pred)
accscore =  accuracy_score(y_test,y_pred)
print("Passive Aggressive Classifier :", accscore)
joblib.dump(model,'fakeNewsDetector2.pkl')



