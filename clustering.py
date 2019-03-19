import gc
import os
import pickle

import numpy as np
import pandas as pd

from ComplaintAnalizer import Clustering
from ComplaintAnalizer import clear_classes
from ComplaintAnalizer import clear_clusters


print('+-----------------------+')
print('| Let\'s go clustering!  |')
print('+-----------------------+')

frames = []
for file in os.listdir('datasets/clustering'):
    frames.append(pd.read_csv(os.path.abspath('datasets/clustering/' + file), header=None))

dataset = pd.concat(frames).values

del frames
gc.collect()

print('Complete reading dataset')

filtered_dataset = []
for i in range(0, len(dataset)):
    if pd.isnull(dataset[i][8]) or dataset[i][7] != 'Проблема':
        continue
    filtered_dataset.append(dataset[i])

dataset = np.array(filtered_dataset)

del filtered_dataset
gc.collect()

print('Complete filtering dataset')

cl = Clustering(max_features=190,
                msg_column=8,
                min_msg_length=20)

cl.stopwords_from_file('resources/complaint_stopwords.txt')

clear_clusters()
clear_classes()

cl.fit_tsne(dataset,
            min_cluster_size=40,
            perplexity=40,
            n_iter=2000,
            learning_rate=500.0,
            n_components=137)

cl.visualize(save_image_to_file=True, show_tsne_res=True)

clusters = cl.clusters_tsne()
with open('resources/clusters.pkl', 'wb') as file:
    pickle.dump(clusters, file)

cl.report('reports/clusters.pdf', show_tsne_res=True)

print('Complete clustering')