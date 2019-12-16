import matplotlib.pyplot as plt
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names


def word_feats(words):
    return dict([(word, True) for word in words])


positive_vocab = ['awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)','happy']
negative_vocab = ['bad', 'terrible', 'useless', 'hate', ':(','evil','sad']
neutral_vocab = ['movie', 'the', 'sound', 'was', 'is', 'actors', 'did', 'know','looking', 'words', 'not','you','u','are','girl','boy','hey','but']

positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]

train_set = negative_features + positive_features + neutral_features

classifier = NaiveBayesClassifier.train(train_set)

# Predict
neg = 0
pos = 0
neu = 0
sentence = input('ENTER THE TEXT: ')
sentence = sentence.lower()
words = sentence.split(' ')
for word in words:
    classResult = classifier.classify(word_feats(word))
    if classResult == 'neg':
        neg = neg + 1
    if classResult == 'pos':
        pos = pos + 1
    if classResult == 'neu':
        neu = neu + 1

p = str(float(pos) / len(words)*100)
print('Positive:' + p)
n= str(float(neg) / len(words)*100)
print('Negative: ' + n)
n2 = str(float(neu) / len(words)*100)
print('Neutral: ' +  n2)

#plotting

labels ='POSITIVE','NEGATIVE','NEUTRAL'
sizes = [p,n,n2]
colors = ['gold','red','lightskyblue']
explode = (0,0.1,0)
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow = True, startangle=100)
plt.axis('equal')
plt.show()