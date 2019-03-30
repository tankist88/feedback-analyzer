import logging
import logging.config
import os

import gc
import numpy as np
import pandas as pd

from ComplaintAnalizer import Clustering
from ComplaintAnalizer import clear_classes
from ComplaintAnalizer import clear_clusters


def main():
    logging.config.fileConfig('resources/logging.conf')
    logger = logging.getLogger("feedback-analyzer.clustering")

    logger.info('+-----------------------+')
    logger.info('| Let\'s go clustering!  |')
    logger.info('+-----------------------+')

    frames = []
    for file in os.listdir('datasets/clustering'):
        if not file.endswith('.csv'):
            continue
        frames.append(pd.read_csv(os.path.abspath('datasets/clustering/' + file), header=None))

    if len(frames) > 0:
        dataset = pd.concat(frames).values

        del frames
        gc.collect()

        logger.info('Complete reading dataset')

        filtered_dataset = []
        for i in range(0, len(dataset)):
            if pd.isnull(dataset[i][8]) or dataset[i][7] != 'Проблема':
                continue
            filtered_dataset.append(dataset[i])

        dataset = np.array(filtered_dataset)

        del filtered_dataset
        gc.collect()

        logger.info('Complete filtering dataset')

        cl = Clustering(max_features=250,
                        msg_column=8,
                        min_msg_length=20)

        cl.stopwords_from_file('resources/complaint_stopwords.txt')

        clear_clusters()
        clear_classes()

        cl.fit_tsne(dataset,
                    min_cluster_size=42,
                    perplexity=40,
                    n_iter=2500,
                    learning_rate=500.0)

        cl.visualize(save_image_to_file=True, show_tsne_res=True)

        clusters = cl.clusters_tsne()

        out_df = pd.DataFrame(columns=['cluster', 'text'])
        out_df['cluster'] = [row[0] for row in clusters]
        out_df['text'] = [row[1] for row in clusters]
        out_df.to_csv("resources/train.csv", index=False, sep=';', encoding='windows-1251')

        cl.report('reports/clusters.pdf', show_tsne_res=True)
    else:
        logger.info('Datasets for clustering not found')

    logger.info('Complete clustering')


if __name__ == "__main__":
    main()
