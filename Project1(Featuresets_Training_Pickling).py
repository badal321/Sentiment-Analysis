import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):#list of classifiers
        self._classifiers=classifiers

    def classify(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v) #The category(pos or neg) assinged by each classifier
        print("Votes:", votes) #(pos,pos,pos,pos,neg)
        return mode(votes)

    def confidence(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
            print(votes)

        choice_votes=votes.count(mode_votes)
        conf=choice_votes/len(votes)
        return conf

short_pos = open("positive.txt","r").read()
short_neg = open("negative.txt","r").read()

documents=[]
allowed_word_types=['J'] #Adjective
all_words=[]

#Documents
for p in short_pos.split('\n'): #Reviews are split by new line
    documents.append( (p ,"pos") )#appending a tuple( review , +ve or -ve)
    words=word_tokenize(p)
    pos=nltk.pos_tag(words)#Applying tags
    for w in pos:
        if w[1][0] in allowed_word_types: #w[1]:Tag, w[1][0]:First letter of the tag
            all_words.append(w[0].lower())

for p in short_neg.split('\n'):
    documents.append( (p,"neg") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)  # Applying tags
    for w in pos:
        if w[1][0] in allowed_word_types:  # w[1]:Tag, w[1][0]:First letter of the tag
            all_words.append(w[0].lower())

pickle_in=open("pickled_algos/documents.pickle",'wb')
pickle.dump(documents,pickle_in)
pickle_in.close()

#All Words
all_words = nltk.FreqDist(all_words) #Arranged according to frequency
word_features = list(all_words.keys())[:5000] #First 5000 words
print("Word Features", word_features[1:10])

save_word_features = open("pickled_algos/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)#Gives 1 if w is in words else 0

    return features
featuresets = [(find_features(review), category) for (review, category) in documents]#list of dictionary
print("Featuresets:", featuresets[0])
random.shuffle(featuresets)
pickle_in=open("pickled_algos/featuresets.pickle","wb")
pickle.dump(featuresets, pickle_in)
pickle_in.close()

#Training and testing data
testing_set = featuresets[10000:]
training_set = featuresets[:10000]

#Original NB
#classifier = nltk.NaiveBayesClassifier.train(training_set)
#print("NLTK Naive Bayes:", (nltk.classify.accuracy(classifier, testing_set))*100)
#classifier.show_most_informative_features(15)

#save_classifier = open("pickled_algos/originalnaivebayes5k.pickle","wb")
#pickle.dump(classifier, save_classifier)
#save_classifier.close()


#Multinomial NB
#MNB_classifier = SklearnClassifier(MultinomialNB())
#MNB_classifier.train(training_set)
#print("MNB_classifier:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

#save_classifier = open("pickled_algos/MNB_classifier5k.pickle","wb")
#pickle.dump(MNB_classifier, save_classifier)
#save_classifier.close()


#Bernouli NB
#BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
#BernoulliNB_classifier.train(training_set)
#print("BernoulliNB:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

#save_classifier = open("pickled_algos/BernoulliNB_classifier5k.pickle","wb")
#pickle.dump(BernoulliNB_classifier, save_classifier)
#save_classifier.close()


#Logistic Regression
#LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
#LogisticRegression_classifier.train(training_set)
#print("LogisticRegression:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

#save_classifier = open("pickled_algos/LogisticRegression_classifier5k.pickle","wb")
#pickle.dump(LogisticRegression_classifier, save_classifier)
#save_classifier.close()


#Linear SVC
#LinearSVC_classifier = SklearnClassifier(LinearSVC())
#LinearSVC_classifier.train(training_set)
#print("LinearSVC:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

#save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle","wb")
#pickle.dump(LinearSVC_classifier, save_classifier)
#save_classifier.close()











