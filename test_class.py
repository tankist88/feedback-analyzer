import pickle
import os

import numpy as np
import pandas as pd

from ComplaintAnalizer import Clustering
from ComplaintAnalizer import Classification
from ComplaintAnalizer import clear_classes
from ComplaintAnalizer import clear_clusters


def get_messages(frames_in):
    dataset_in = pd.concat(frames_in).values
    filtered_dataset_in = []
    for index in range(0, len(dataset_in)):
        if pd.isnull(dataset_in[index][8]) or dataset_in[index][7] != 'Проблема':
            continue
        filtered_dataset_in.append(dataset_in[index])
    dataset_in = np.array(filtered_dataset_in)
    
    only_msg = [row[8] for row in dataset_in]
    only_msg = np.array(only_msg).reshape(len(only_msg), 1)
    return only_msg


def fill_db(classifier_in, frames_in, date_in):
    msgs = get_messages(frames_in)
    if len(msgs) > 0:
        classifier_in.predict_backup(messages=msgs,
                                     date=date_in,
                                     clear_db=False,
                                     threshold=0.4)


# ============== Clustering ================
dataset1 = pd.read_csv('datasets/data_20150923.csv')

frames = [dataset1]

dataset = pd.concat(frames).values

filtered_dataset = []
for i in range(0, len(dataset)):
    if pd.isnull(dataset[i][8]) or dataset[i][7] != 'Проблема':
        continue
    filtered_dataset.append(dataset[i])

dataset = np.array(filtered_dataset)

cl = Clustering(max_features=190,
                msg_column=8,
                min_msg_length=20)
cl.stopwords_from_file('complaint_stopwords.txt')
clear_clusters()
clear_classes()

# from ComplaintAnalizer import text_preprocessing
# corpus, orig = text_preprocessing(dataset, 8, 20, cl.get_stop_words_set())
# from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer(max_features=190)
# X = cv.fit_transform(corpus).toarray()
#
# from sklearn.decomposition import PCA
# pca = PCA(n_components=137)
# x_pca = pca.fit_transform(X)
# ratio = pca.explained_variance_ratio_
# print('Cumulative explained variation for principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))


cl.fit_tsne(dataset,
            min_cluster_size=40,
            perplexity=40,
            n_iter=2000,
            learning_rate=500.0,
            n_components=137)
cl.visualize(save_image_to_file=True, show_tsne_res=True)
clusters = cl.clusters()
cl.report('reports/clusters.pdf')

with open('clusters.pkl', 'wb') as fout:
    pickle.dump(clusters, fout)

# ============== Classification training ================
with open('clusters.pkl', 'rb') as file:  
    clusters = pickle.load(file)

classifier = Classification()
classifier.stopwords_from_file('complaint_stopwords.txt')
classifier.fit(dataset=clusters,
               batch_size=5,
               nb_epoch=50,
               min_msg_length=20,
               save_image_to_file=True)

# ============== Classification predicting ================
classifier = Classification()
classifier.stopwords_from_file('complaint_stopwords.txt')
clear_classes()

files = [
         'datasets/data_2012-04-01.csv',
         'datasets/data_2012-06-01.csv',
         'datasets/data_2012-07-01.csv',
         'datasets/data_2012-08-01.csv',
         'datasets/data_2012-09-01.csv',
         'datasets/data_2012-10-01.csv',
         'datasets/data_2012-11-01.csv',
         'datasets/data_2012-12-01.csv',
         'datasets/data_2013-01-01.csv',
         'datasets/data_2013-02-01.csv',
         'datasets/data_2013-03-01.csv',
         'datasets/data_2013-04-01.csv',
         'datasets/data_2013-05-01.csv',
         'datasets/data_2013-06-01.csv',
         'datasets/data_2013-07-01.csv',
         'datasets/data_2013-08-01.csv',
         'datasets/data_2013-09-01.csv',
         'datasets/data_2013-10-01.csv',
         'datasets/data_2013-11-01.csv',
         'datasets/data_2013-12-01.csv',
         'datasets/data_2014-01-01.csv',
         'datasets/data_2014-02-01.csv',
         'datasets/data_2014-03-01.csv',
         'datasets/data_2014-04-01.csv',
         'datasets/data_2014-05-01.csv',
         'datasets/data_2014-06-01.csv',
         'datasets/data_2014-07-01.csv',
         'datasets/data_2014-08-01.csv',
         'datasets/data_2014-09-01.csv',
         'datasets/data_2014-10-01.csv',
         'datasets/data_2014-11-01.csv',
         'datasets/data_2014-12-01.csv',
         'datasets/data_2015-01-01.csv',
         'datasets/data_2015-02-01.csv',
         'datasets/data_2015-03-01.csv',
         'datasets/data_2015-04-01.csv',
         'datasets/data_2015-05-01.csv',
         'datasets/data_2015-06-01.csv',
         'datasets/data_2015-07-01.csv',
         'datasets/data_2015-08-01.csv',
         'datasets/data_2015-09-01.csv']

for file in files:
    ds = pd.read_csv(os.path.abspath(file), header=None)
    frames = [ds]
    date = file.split(sep='_')[1].split(sep='.')[0]
    fill_db(classifier, frames, date)

classifier.classes_report('reports/dynamics.pdf')
