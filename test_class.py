import pickle
import os

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.patheffects as PathEffects

from ComplaintAnalizer import Clustering
from ComplaintAnalizer import Classification
from ComplaintAnalizer import clear_classes
from ComplaintAnalizer import clear_clusters


def eval_clustering(dataset_in, max_features, som_sigma, som_learning_rate):
    cl_in = Clustering(max_features=max_features,
                       som_threshold=0.55,
                       som_sigma=som_sigma,
                       som_learning_rate=som_learning_rate,
                       msg_column=8,
                       min_msg_length=20)
    cl_in.stopwords_from_file('complaint_stopwords.txt')
    cl_in.fit(dataset_in)
    return cl_in.mean_density()


def model_densities(dataset_in):
    max_features_arr = np.array([250])
    som_sigma_arr = np.array([1.0, 1.3])
    som_learning_rate_arr = np.array([0.5, 0.7])

    densities = []
    
    for index in range(0, len(max_features_arr)):
        for j in range(0, len(som_sigma_arr)):
            for k in range(0, len(som_learning_rate_arr)):
                density = eval_clustering(dataset_in,
                                          max_features_arr[index],
                                          som_sigma_arr[j],
                                          som_learning_rate_arr[k])
                row = list()
                row.append(density)
                row.append(max_features_arr[index])
                row.append(som_sigma_arr[j])
                row.append(som_learning_rate_arr[k])
                print("---------------------")
                print(row)
                print("---------------------")
                densities.append(np.array(row, dtype='float64'))
    return np.array(densities, dtype='float64')


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
                                     threshold=0.00000001)


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

# densities = model_densities(dataset)

cl = Clustering(max_features=190,
                som_threshold=0.35,
                som_sigma=1.3,
                som_learning_rate=0.7,
                msg_column=8,
                min_msg_length=20)
cl.stopwords_from_file('complaint_stopwords.txt')
clear_clusters()
clear_classes()

from ComplaintAnalizer import text_preprocessing
corpus, orig = text_preprocessing(dataset, 8, 30, cl.get_stop_words_set())
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=190)
X = cv.fit_transform(corpus).toarray()

from sklearn.decomposition import PCA
pca = PCA(n_components=99)
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1500)
tsne_results = tsne.fit_transform(X_pca)

df = pd.DataFrame(columns=['X', 'Y'])
df['X'] = tsne_results[:, 0]
df['Y'] = tsne_results[:, 1]

from hdbscan import HDBSCAN
clustering = HDBSCAN(min_cluster_size=29)
cl_X = df.values
y_pred = clustering.fit_predict(cl_X)
n_clusters = y_pred.max() + 1

print('n_clusters: ' + str(n_clusters))


import matplotlib.pyplot as plt
plt.scatter(tsne_results[:, 0],
            tsne_results[:, 1],
            s=25,
            c='black',
            edgecolors='none',
            label='Clusters ')
plt.title('Clusters')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

available_colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'orange',
                    'yellow', 'brown', 'grey', 'navy', 'purple', 'lightcoral',
                    'lime', 'steelblue', 'indigo', 'olive', 'khaki', 'crimson',
                    'slateblue', 'gold', 'darkseagreen', 'violet', 'black']
plt.figure(figsize=(7, 6))

clusters_codes = pd.DataFrame(y_pred, columns=['cl'])['cl'].unique()

for i in range(0, n_clusters):
    cl_code = clusters_codes[i]
    if cl_code < 0:
        continue
    plt.scatter(cl_X[y_pred == cl_code, 0],
                cl_X[y_pred == cl_code, 1],
                s=25,
                c=available_colors[i],
                edgecolors='none',
                label='Cluster ' + str(cl_code))
plt.title('Clusters')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig("test.png")
plt.show()



cl.fit(dataset)
cl.visualize(save_image_to_file=True)
cl.som_mappings()
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
               batch_size=1,
               nb_epoch=25,
               min_msg_length=20)
cm = classifier.get_confusion_matrix()

# ============== Classification predicting ================
classifier = Classification()
classifier.stopwords_from_file('complaint_stopwords.txt')
clear_classes()

files = ['datasets/data_2012-04-01.csv',
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
