import os

import pandas as pd

import logging
import logging.config

from ComplaintAnalizer import Classification


def main():
    logging.config.fileConfig('resources/logging.conf')
    logger = logging.getLogger("feedback-analyzer.train_classifier")
    
    logger.info('+-----------------------+')
    logger.info('| Let\'s go train!       |')
    logger.info('+-----------------------+')

    clusters = pd.read_csv(
        os.path.abspath('resources/train.csv'),
        delimiter=';',
        encoding='windows-1251'
    ).values
    
    classifier = Classification()
    classifier.stopwords_from_file('resources/complaint_stopwords.txt')
    classifier.fit(dataset=clusters,
                   batch_size=5,
                   nb_epoch=50,
                   min_msg_length=20,
                   save_image_to_file=True)
    
    logger.info('Complete train')


if __name__ == "__main__":
    main()
