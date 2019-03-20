import os

import numpy as np
import pandas as pd

from ComplaintAnalizer import Classification
from ComplaintAnalizer import clear_classes


def get_messages(dataset_in):
    filtered_dataset_in = []
    for index in range(0, len(dataset_in)):
        if pd.isnull(dataset_in[index][8]) or dataset_in[index][7] != 'Проблема':
            continue
        filtered_dataset_in.append(dataset_in[index])
    dataset_in = np.array(filtered_dataset_in)

    only_msg = [row[8] for row in dataset_in]
    only_msg = np.array(only_msg).reshape(len(only_msg), 1)
    return only_msg


def fill_db(classifier_in, dataset_in, date_in, ths):
    msgs = get_messages(dataset_in)
    if len(msgs) > 0:
        classifier_in.predict_backup(messages=msgs,
                                     date=date_in,
                                     clear_db=False,
                                     threshold=ths)


print('+-----------------------+')
print('| Let\'s go evaluate!    |')
print('+-----------------------+')

classifier = Classification()
classifier.stopwords_from_file('resources/complaint_stopwords.txt')
clear_classes()

for file in os.listdir('datasets/classification'):
    ds = pd.read_csv(os.path.abspath('datasets/classification/' + file), header=None)
    date_str = file.split(sep='_')[1].split(sep='.')[0]
    fill_db(classifier, ds.values, date_str, 0.4)

classifier.classes_report('reports/dynamics.pdf', 3)

print('Complete evaluate')
