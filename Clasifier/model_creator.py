import csv
from sklearn.feature_extraction import DictVectorizer
import logging
import random
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import *
from sklearn import linear_model
from sklearn.datasets.base import Bunch
from sklearn import metrics
import pickle
import datetime

class Model(object):

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    log.addHandler(handler)

    def __init__(self, classifier, vectorizer, actual_values, prediction, f1_score, name= None):
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.actual_values = actual_values
        self.prediction = prediction
        self.f1_score = f1_score
        self.name = name
        self.save()

    def save(self):
        prefix1 = "f1_{}_date_{}_classifier_{}".format(self.f1_score, datetime.datetime.now(), self.name)
        prefix2 = "f1_{}_date_{}_vectorizer_{}".format(self.f1_score, datetime.datetime.now(), self.name)
        self.log.info("model prefix- {}".format(prefix1))
        pickle.dump( self.classifier, open( prefix1, "wb" ))
        pickle.dump(self.vectorizer, open(prefix2, "wb"))

    def __repr__(self):
         print self.f1_score

class ModelCreator(object):

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    log.addHandler(handler)

    accident_contents = []
    non_accident_contents = []
    label_human_desc = ["Accidents", "Non Accidents"]
    training_data = None
    test_data = None
    test_size = 0.05
    f1_score = 0

    def __init__(self, accident_contents_list, non_accident_contents_list):
        self.accident_contents = accident_contents_list
        self.non_accident_contents = non_accident_contents_list
        self.shuffle_data()

    def shuffle_data(self):
        self.log.info("Shuffling all training data...")
        if self.accident_contents and self.non_accident_contents:
            self.combined_data = self.accident_contents + self.non_accident_contents
            self.combined_labels = [1] * len(self.accident_contents) + [0] * len(self.non_accident_contents)

            combined = zip(self.combined_data, self.combined_labels)
            random.shuffle(combined)
            self.log.info("Done Shuffling...")

            self.combined_data[:], self.combined_labels[:] = zip(*combined)
        else:
            raise Exception("No content found")

    def process_training_data(self):
        self.log.info("Processing training data")
        all_data = self.combined_data
        all_labels = self.combined_labels
        length = int(len(all_data) * (1 - self.test_size))
        training_data = all_data[:length]
        test_data = all_data[length:]
        training_labels = all_labels[:length]
        self.log.info("length {}".format(length))
        test_labels = all_labels[length:]

        self.training_data = Bunch(
            data=training_data,
            target_names=self.label_human_desc,
            target=training_labels,
            DESCR="Linqia blog post classification training dataset")

        self.test_data = Bunch(
             data=test_data,
             target_names=self.label_human_desc,
             target=test_labels,
             DESCR="Linqia blog post classification test dataset")

    def learn(self):
        self.log.info("Creating the model")
        vectorizer = DictVectorizer()
        self.log.info("Creating vectorizer")
        features = vectorizer.fit_transform(self.training_data.data)


        if self.name == "SGDClassifier":
            self.log.info("Creating SGD classifier")
            clf = linear_model.SGDClassifier().fit(features, self.training_data.target)
        elif self.name == "NaiveBayes":
            self.log.info("Creating NaiveBayes classifier")
            clf = MultinomialNB(alpha=.90).fit(features, self.training_data.target)

        self.vectorizer = vectorizer
        self.classifier = clf

    def cross_validate(self):
        self.log.info("Cross validating")
        vectors_test = self.vectorizer.transform(self.test_data.data)
        self.prediction = self.classifier.predict(vectors_test)
        self.f1_score = metrics.f1_score(self.test_data.target, self.prediction)
        self.log.info("f1-score:   %0.3f" % self.f1_score)
        report = metrics.classification_report(self.test_data.target, self.prediction,
            target_names=self.label_human_desc)
        self.log.info(report)
        print(report)

    def run(self, name=None):
        self.name = name
        self.process_training_data()
        self.learn()
        self.cross_validate()

        #return Model(self.classifier, self.vectorizer,None, None, self.f1_score, self.name)
        return Model(self.classifier, self.vectorizer,self.test_data.target, self.prediction, self.f1_score, self.name)

if __name__ == "__main__":

        accident_content_list = list(csv.DictReader(open('/Users/ranjitkhanuja/PycharmProjects/mining/data/training_data_fatality.csv', 'rU')))
        non_accident_content_list = list(csv.DictReader(open('/Users/ranjitkhanuja/PycharmProjects/mining/data/training_data_noaccidents.csv', 'rU')))

        builder = ModelCreator(accident_contents_list=accident_content_list, non_accident_contents_list=non_accident_content_list)

        model_sgd = builder.run(name= "SGDClassifier")
        model_sgd.f1_score

        model_nb = builder.run(name= "NaiveBayes")
        model_nb.f1_score