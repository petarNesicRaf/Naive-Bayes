import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

nltk.download('stopwords')


class MultinomialNaiveBayes:
  #nisam imao dovoljno vremena da zavrsim ovaj zadatak tako da je dosta toga prelepljeno  
  def __init__(self, nb_classes, nb_words, pseudocount):
    self.nb_classes = nb_classes
    self.nb_words = nb_words
    self.pseudocount = pseudocount
  
  def fit(self, X, Y):
    nb_examples =X.shape[0]
    self.priors = np.bincount(Y) / nb_examples

    print('Priors:')
    print(self.priors)

    occs = np.zeros((self.nb_classes, self.nb_words))
    for i in range(nb_examples):
      c = Y[i]
      for w in range(self.nb_words):
        cnt = X[i][w]
        occs[c][w] +=cnt
    print('Occurences:')
    print(occs)
    

    self.like = np.zeros((self.nb_classes, self.nb_words))
    for c in range(self.nb_classes):
      for w in range(self.nb_words):
        up = occs[c][w] + self.pseudocount
        down = np.sum(occs[c]) + self.nb_words * self.pseudocount
        self.like[c][w] = up/down
    print('Likelihoods:')
    print(self.like)

  def predict(self, bow):
    probs = np.zeros(self.nb_classes)
    for c in range(self.nb_classes):
      prob = np.log(self.priors[c])
      for w in range(self.nb_words):
        cnt = bow[w]
        prob += cnt * np.log(self.like[c][w])
      probs[c] = prob
    print('\"Probabilites\" for a test BoW (with log):')
    print(probs)
    prediction = np.argmax(probs)
    return prediction
  
  def predict_multiply(self, bow):
    probs = np.zeros(self.nb_classes)
    for c in range(self.nb_classes):
      prob = self.priors[c]
      for w in range(self.nb_words):
        cnt = bow[w]
        prob *= self.like[c][w] ** cnt
      probs[c] = prob
    print('\"Probabilities\" for a test BoW (without log):')
    print(probs)  
    prediction = np.argmax(probs)
    return prediction



def clean_tweet(text):
    
    text = text.lower()
    #url i @
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@[^\s]+', '', text)
    #izbaci sve sto nije reg
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    #split po recima
    tokens = nltk.word_tokenize(text)
    #stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    #stemer
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    #spajanje
    cleaned_text = ' '.join(stemmed_tokens)
    
    return cleaned_text


df = pd.read_csv('disaster-tweets.csv')

dic = dict()
dic['x'] = df['text'].tolist()
dic['y'] = df['target'].tolist()

x_clean = []
for i in range(len(dic['x'])):
    x_clean.append(clean_tweet(dic['x'][i]))


# Calculate the index to split the data at
split_index = int(0.8 * len(df))

#splitujemo dataset
data = dict()
data['x'] = x_clean

train_x = data['x'][:split_index]
train_y = dic['y'][:split_index]

test_x = data['x'][split_index:]


all_train_tweets = train_x
print('corpus')
print(all_train_tweets)


#test clean


#pravimo bow matricu semplovi x reci
vectorizer = CountVectorizer()
x_bow = vectorizer.fit_transform(train_x).toarray()
#sve unikatne reci
feature_names = list(vectorizer.vocabulary_.keys())
print(feature_names)

Y = np.asarray(train_y)
X = np.asarray(x_bow)

test_bow = vectorizer.fit_transform(test_x).toarray()

mnb = MultinomialNaiveBayes(nb_classes=2, nb_words=X.shape[1], pseudocount=1)
mnb.fit(X, Y)
#prediction = mnb.predict(test_bow)
#print('predicted class with log ', prediction )

