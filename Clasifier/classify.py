__author__ = 'ranjitkhanuja'
import pickle
import csv

classifier1 = pickle.load(open('/Users/ranjitkhanuja/PycharmProjects/mining/f1_0.78527607362_date_2014-11-16 09:40:03.485742_classifier_NaiveBayes','rb'))
vectorizer1 = pickle.load(open('/Users/ranjitkhanuja/PycharmProjects/mining/f1_0.78527607362_date_2014-11-16 09:40:03.485759_vectorizer_NaiveBayes','rb'))

classifier2 = pickle.load(open('/Users/ranjitkhanuja/PycharmProjects/mining/f1_0.892655367232_date_2014-11-16 09:40:03.073156_classifier_SGDClassifier','rb'))
vectorizer2 = pickle.load(open('/Users/ranjitkhanuja/PycharmProjects/mining/f1_0.892655367232_date_2014-11-16 09:40:03.073180_vectorizer_SGDClassifier','rb'))


test_content_list = list(csv.DictReader(open('/Users/ranjitkhanuja/PycharmProjects/mining/data/test_data_newmine.csv', 'rU')))
print len(test_content_list)

mine_ids =[]
for val in test_content_list:
    mine_ids.append(val.get('MINE_ID'))
print mine_ids
transform = vectorizer1.transform(test_content_list)
i=vectorizer1.inverse_transform(transform)

prediction1=classifier1.predict(transform)

#print prediction1

transform = vectorizer2.transform(test_content_list)
i=vectorizer2.inverse_transform(transform)

prediction2=classifier2.predict(transform)

#print prediction2

import numpy as np
prediction_sum = np.array(prediction1) + np.array(prediction2)
#print prediction_sum

index=0
for val in prediction_sum:
    if val ==2:
        print mine_ids[index]
    index+=1