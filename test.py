import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# nltk.download()
porter = SnowballStemmer('english')
myTfidf = TfidfVectorizer()

def stemIt(text):
    return [porter.stem(word) for word in text]

def stopIt(text):
    return [word for word in text if len(word) > 2]

data = pd.read_csv("True.csv")
data = data.drop(['title','subject','date'], axis=1)
data= data['text'][100]
# tfidf_test = myTfidf.transform(data)
data = word_tokenize(data)
# print(data)

data = stemIt(data)
data = stopIt(data)
vocabulary = list(set(data))
data = ' '.join(data)
corpus = [data]
# print(data)
# data = list(set(stopIt(data)))
# print(data)

pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),('tfid', TfidfTransformer())]).fit(corpus)
pipe['count'].transform(corpus).toarray()
input=[[pipe.transform(corpus).shape,pipe['tfid'].idf_]]
# print(pipe)

model = joblib.load('fakeNewsDetector2.pkl')
print(model.predict(input))


