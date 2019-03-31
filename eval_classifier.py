import os

import logging
import logging.config

import numpy as np
import pandas as pd

from ComplaintAnalizer import Classification


def get_messages(dataset):
    filtered_dataset = []
    for index in range(0, len(dataset)):
        if pd.isnull(dataset[index][8]) or dataset[index][7] != 'Проблема':
            continue
        filtered_dataset.append(dataset[index])
    filtered_dataset = np.asarray(filtered_dataset)

    only_msg = [row[8] for row in filtered_dataset]
    only_msg = np.array(only_msg).reshape(len(only_msg), 1)
    return only_msg


def fill_db(classifier, dataset, date, ths):
    msgs = get_messages(dataset)
    if len(msgs) > 0:
        classifier.predict_backup(
            messages=msgs,
            date=date,
            clear_db=False,
            threshold=ths)


def main():
    logging.config.fileConfig('resources/logging.conf')
    logger = logging.getLogger("feedback-analyzer.eval_classifier")
    
    logger.info('+-----------------------+')
    logger.info('| Let\'s go evaluate!    |')
    logger.info('+-----------------------+')

    classifier = Classification()
    classifier.stopwords_from_file('resources/complaint_stopwords.txt')

    new_file_count = 0
    for file in os.listdir('datasets/classification'):
        if not file.endswith('.csv'):
            continue

        logger.info('File: {:03d}'.format(new_file_count + 1))

        ds = pd.read_csv(os.path.abspath('datasets/classification/' + file), header=None)
        date_str = file.split(sep='_')[1].split(sep='.')[0]
        fill_db(classifier, ds.values, date_str, 0.4)
        new_file_count += 1

    if new_file_count > 0:
        classifier.classes_report('reports/dynamics.pdf', 3)
    else:
        logger.info('New files for evaluating not found')

    logger.info('Complete evaluate')


if __name__ == "__main__":
    main()
