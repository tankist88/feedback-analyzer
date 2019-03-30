import datetime
import logging.config
import os
import pickle
import re
import sqlite3

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pylab as pyb
import seaborn as sns
from hdbscan import HDBSCAN
from jinja2 import Environment, FileSystemLoader
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from matplotlib import cm
from minisom import MiniSom
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from pandas import isnull
from pymorphy2 import MorphAnalyzer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from weasyprint import HTML
from wordcloud import WordCloud

TARGET_DIR = 'target'
RESOURCE_DIR = 'resources'
DB_NAME = 'complaint_stat.db'

nltk.download('stopwords')

logging.config.fileConfig(RESOURCE_DIR + '/logging.conf')
logger = logging.getLogger("feedback-analyzer.ComplaintAnalizer")


def get_prepared_word_array(text):
    line = re.sub('[^а-яА-Я]', ' ', text)
    line = line.lower()
    word_array = line.split()
    return word_array


def text_preprocessing(dataset, msg_column, min_msg_length, stop_words_set):
    corpus = []
    orig = []
    for i in range(0, len(dataset)):
        text = dataset[i, msg_column]
        if isnull(text) or len(text) < min_msg_length:
            continue
        review = get_prepared_word_array(text)
        ss = SnowballStemmer('russian')
        filtered_review = []
        for word in review:
            if word not in stop_words_set:
                filtered_review.append(ss.stem(word))
        review = ' '.join(np.array(filtered_review))
        corpus.append(review)
        orig.append(text)
    return corpus, orig


def db_init():
    conn = sqlite3.connect(RESOURCE_DIR + '/' + DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS COMPLAINT_CLASSES 
        (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            CL_NUM INTEGER NOT NULL, 
            CL_SIZE INTEGER NOT NULL, 
            CL_DATE TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS COMPLAINT_MESSAGES 
        (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            CL_ID INTEGER NOT NULL, 
            CL_NUM INTEGER NOT NULL, 
            CL_MSG TEXT NOT NULL, 
            FOREIGN KEY(CL_ID) REFERENCES COMPLAINT_CLASSES(ID)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS CLASSES_LOGO 
        (
            CL_NUM INTEGER PRIMARY KEY, 
            CL_LOGO BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def clear_classes():
    db_init()
    conn = sqlite3.connect(RESOURCE_DIR + '/' + DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM COMPLAINT_CLASSES")
    cursor.execute("DELETE FROM COMPLAINT_MESSAGES")
    conn.commit()
    conn.close()
    logger.info('All classes deleted')


def clear_clusters():
    db_init()
    conn = sqlite3.connect(RESOURCE_DIR + '/' + DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM CLASSES_LOGO")
    conn.commit()
    conn.close()
    logger.info('All clusters deleted')


def get_file_path(filename, directory, overwrite=True):
    if not os.path.isdir(os.path.abspath(directory)):
        os.mkdir(os.path.abspath(directory))
    if overwrite:
        return os.path.abspath(directory + '/' + filename + '.png')
    else:
        return free_file_name(os.path.abspath(directory + '/' + filename), 'png')


def free_file_name(filename, ext):
    new_filename = str(filename) + '.' + str(ext)
    index = 1
    while os.path.isfile(new_filename):
        new_filename = str(filename) + '_' + str(index) + '.' + str(ext)
        index = index + 1
    return new_filename


def get_next_cluster_num():
    db_init()

    conn = sqlite3.connect(RESOURCE_DIR + '/' + DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT CL_NUM FROM CLASSES_LOGO ORDER BY CL_NUM DESC")
    result = cursor.fetchone()
    conn.commit()
    conn.close()

    if pd.isnull(result):
        return int(0)
    else:
        return int(result[0]) + 1


def insert_cluster_logo(num):
    db_init()

    filepath = os.path.abspath(TARGET_DIR + '/cl_' + str(num) + '.png')

    with open(filepath, 'rb') as f:
        ablob = f.read()

    conn = sqlite3.connect(RESOURCE_DIR + '/' + DB_NAME)

    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO CLASSES_LOGO VALUES(?, ?)", (int(num), sqlite3.Binary(ablob)))
        conn.commit()
    except Exception as e:
        logger.info('Exception while insert logo to db')

    conn.close()


def insert_class_row(num, size, date):
    conn = sqlite3.connect(RESOURCE_DIR + '/' + DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO COMPLAINT_CLASSES VALUES(null, ?, ?, ?)", (num, size, date))
    conn.commit()
    cursor.execute("SELECT ID FROM COMPLAINT_CLASSES ORDER BY ID DESC")
    cl_id = cursor.fetchone()
    conn.close()
    return int(cl_id[0])


def insert_msg_row(cl_id, num, msg):
    conn = sqlite3.connect(RESOURCE_DIR + '/' + DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO COMPLAINT_MESSAGES VALUES(null, ?, ?, ?)", (cl_id, num, msg))
    conn.commit()
    conn.close()


def get_all_classes():
    conn = sqlite3.connect(RESOURCE_DIR + '/' + DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM COMPLAINT_CLASSES ORDER BY date(CL_DATE) ASC")
    results = cursor.fetchall()
    conn.close()
    return results


def get_messages_by_class_id(cl_id):
    conn = sqlite3.connect(RESOURCE_DIR + '/' + DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT CL_MSG FROM COMPLAINT_MESSAGES WHERE CL_ID=?", (cl_id,))
    results = cursor.fetchall()
    conn.close()
    return np.array(([row[0] for row in results]))


def extract_all_logos():
    conn = sqlite3.connect(RESOURCE_DIR + '/' + DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT CL_NUM, CL_LOGO FROM CLASSES_LOGO")
    results = cursor.fetchall()
    conn.close()

    if len(results) > 0:
        for result in results:
            filename = get_file_path('cl_' + str(result[0]) + '.png', directory=TARGET_DIR)
            with open(filename, 'wb') as f:
                f.write(result[1])


class Clustering:
    def __init__(self, 
                 max_features=250,
                 msg_column=0,
                 min_msg_length=30,
                 period_start='____________',
                 period_end='____________',
                 overwrite=True):
        self.max_features = max_features
        self.msg_column = msg_column
        self.min_msg_length = min_msg_length
        self.period_start = period_start
        self.period_end = period_end
        self.overwrite = overwrite
        self.is_som_mappings_set = False
        self.stop_words_set = set(stopwords.words('russian'))

    def get_original_x_rows_for_cluster(self, cluster_num, limit=-1):
        if limit > 0:
            logger.info('--- cluster ' + str(cluster_num) + '---')
        
        cluster_lines = []

        cluster = self.index_array[self.y == cluster_num]

        cluster = cluster[np.argsort(cluster[:, 2])]

        if 0 < limit < len(cluster):
            cluster_length = limit
        else:
            cluster_length = len(cluster)
            
        if self.is_som_mappings_set:
            for i in range(0, cluster_length):
                if limit > 0:
                    logger.info('cluster[' + str(i) + ']: ' + str(cluster[i][2]))
                array_of_arrays = self.mappings[(cluster[i][0], cluster[i][1])]
                for j in range(0, len(array_of_arrays)):
                    cluster_lines.append(array_of_arrays[j])
            if len(cluster_lines) > 0:
                cluster_lines = self.sc.inverse_transform(np.array(cluster_lines))
            return cluster_lines
        else:
            for i in range(0, len(self.y_pred)):
                if self.y_pred[i] == cluster_num:
                    cluster_lines.append(self.X[i])
            if limit > 0:
                rnd_indexes = np.random.choice(len(cluster_lines), cluster_length)
                limited_lines = []
                for rnd_idx in range(len(rnd_indexes)):
                    limited_lines.append(cluster_lines[rnd_idx])
                return limited_lines
            else:
                return cluster_lines
    
    def fit_som(self, 
                dataset, 
                som_threshold=0.5, 
                som_size=100, 
                som_sigma=1.0, 
                som_learning_rate=0.5):
        self.corpus, self.orig = text_preprocessing(dataset, 
                                                    self.msg_column, 
                                                    self.min_msg_length, 
                                                    self.stop_words_set)
        
        logger.info("corpus length: " + str(len(self.corpus)))
            
        fname = \
            RESOURCE_DIR + '/som_' + \
            str(self.max_features) + '_' + \
            str(som_size) + '_' + \
            str(som_sigma).replace(".", "_") + '_' + \
            str(som_learning_rate).replace(".", "_")
        
        if not self.overwrite:
            fname = free_file_name(fname, 'pkl')
        else:
            fname = fname + '.pkl'

        logger.info("Using SOM model file: " + str(fname))
        
        if not os.path.isfile(fname):
            cv = CountVectorizer(max_features=self.max_features)
            self.X = cv.fit_transform(self.corpus).toarray()
            self.sc = MinMaxScaler(feature_range=(0, 1))
            self.X_scale = self.sc.fit_transform(self.X)
            self.som = MiniSom(x=som_size,
                               y=som_size,
                               input_len=self.max_features,
                               sigma=som_sigma,
                               learning_rate=som_learning_rate)
            self.som.train_batch(data=self.X_scale, num_iteration=len(self.X_scale))
            with open(fname, 'wb') as file:
                pickle.dump((cv, self.sc, self.som), file)
        else:
            with open(fname, 'rb') as file:  
                cv, self.sc, self.som = pickle.load(file)
            self.X = cv.fit_transform(self.corpus).toarray()
            self.X_scale = self.sc.fit_transform(self.X)
        
        logger.info('X rows: ' + str(len(self.X)))
        logger.info('X cols: ' + str(len(self.X[0])))
        
        distance_map = self.som.distance_map()
        
        indexes_coords = []
        indexes_dist = []
        for i in range(0, len(distance_map)):
            for j in range(0, len(distance_map[i])):
                if distance_map[i, j] < som_threshold:
                    indexes_coords.append(i)
                    indexes_coords.append(j)
                    indexes_dist.append(distance_map[i, j])
        
        coord_array = np.array(indexes_coords).reshape(int(len(indexes_coords)/2), 2)
        dist_array = np.array(indexes_dist).reshape(int(len(indexes_dist)), 1)

        clustering = HDBSCAN(min_cluster_size=5)
        y_pred = clustering.fit_predict(coord_array)
        
        next_cluster_num = get_next_cluster_num()
        
        self.y = []
        filtered_coords = []
        filtered_dists = []
        for i in range(0, len(y_pred)):
            if y_pred[i] < 0:
                continue
            filtered_coords.append(coord_array[i])
            filtered_dists.append(dist_array[i])
            self.y.append(int(y_pred[i]) + int(next_cluster_num))
        
        self.y = np.array(self.y, dtype='int')
        
        logger.info('Next cluster num: ' + str(next_cluster_num))
        
        clusters_codes = pd.DataFrame(self.y, columns=['cl'])['cl'].unique()
        
        self.n_clusters = len(clusters_codes)
        
        coord_array = np.array(filtered_coords)
        dist_array = np.array(filtered_dists)
        
        self.index_array = np.concatenate((coord_array, dist_array), axis=1)
        
        logger.info('Cluster codes (' + str(self.n_clusters) + '):')
        logger.info(clusters_codes)
        
    def fit_tsne(self, 
                 dataset, 
                 min_cluster_size=55, 
                 perplexity=40, 
                 n_iter=2500, 
                 learning_rate=700.0, 
                 n_components=107):
        self.corpus, self.orig = text_preprocessing(dataset, 
                                                    self.msg_column, 
                                                    self.min_msg_length, 
                                                    self.stop_words_set)
        
        logger.info("corpus length: " + str(len(self.corpus)))
            
        fname = \
            RESOURCE_DIR + '/tsne_' + \
            str(self.max_features) + '_' + \
            str(n_iter) + '_' +  \
            str(n_components) + '_' + \
            str(perplexity).replace(".", "_") + '_' + \
            str(learning_rate).replace(".", "_")
        
        if not self.overwrite:
            fname = free_file_name(fname, 'pkl')
        else:
            fname = fname + '.pkl'

        logger.info("Using t-SNE model file: " + str(fname))
        
        if not os.path.isfile(fname):
            cv = CountVectorizer(max_features=self.max_features)
            self.X = cv.fit_transform(self.corpus).toarray()

            pca = PCA(n_components=n_components)
            x_pca = pca.fit_transform(self.X)
            logger.info('Cumulative explained variation for principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))

            tsne = TSNE(n_components=2, 
                        verbose=1, 
                        perplexity=perplexity, 
                        n_iter=n_iter, 
                        learning_rate=learning_rate)
            self.tsne_results = tsne.fit_transform(x_pca)            
            with open(fname, 'wb') as file:
                pickle.dump((cv, pca, self.tsne_results), file)
        else:
            with open(fname, 'rb') as file:  
                cv, pca, self.tsne_results = pickle.load(file)
            self.X = cv.fit_transform(self.corpus).toarray()
            self.x_pca = pca.fit_transform(self.X)
        
        logger.info('X rows: ' + str(len(self.X)))
        logger.info('X cols: ' + str(len(self.X[0])))
        
        df = pd.DataFrame(columns=['X', 'Y'])
        df['X'] = self.tsne_results[:, 0]
        df['Y'] = self.tsne_results[:, 1]

        tsne_values = df.values

        clustering = HDBSCAN(min_cluster_size=min_cluster_size)
        self.y_pred = clustering.fit_predict(tsne_values)
        
        next_cluster_num = get_next_cluster_num()
        
        self.y = []
        filtered_values = []
        for i in range(0, len(self.y_pred)):
            if self.y_pred[i] < 0:
                continue
            filtered_values.append(tsne_values[i])
            self.y.append(int(self.y_pred[i]) + int(next_cluster_num))
        
        self.y = np.array(self.y, dtype='int')
        
        logger.info('Next cluster num: ' + str(next_cluster_num))
        
        clusters_codes = pd.DataFrame(self.y, columns=['cl'])['cl'].unique()
        
        self.n_clusters = len(clusters_codes)
        
        filtered_values = np.array(filtered_values)
        
        self.index_array = np.concatenate((filtered_values, np.zeros((len(filtered_values), 1), dtype='float')), axis=1)
        
        logger.info('Cluster codes (' + str(self.n_clusters) + '):')
        logger.info(clusters_codes)
    
    def get_index_array(self):
        return self.index_array
        
    def som_mappings(self):
        self.mappings = self.som.win_map(self.X_scale)
        self.is_som_mappings_set = True
    
    def get_clusters_number(self):
        return self.n_clusters
    
    def get_corpus(self):
        return self.corpus
    
    def mean_density(self):
        density_list = []
        for i in range(0, len(self.index_array)):
            density_list.append(self.index_array[i, 2])
        return np.array(density_list, dtype='float64').mean()
    
    def visualize(self, save_image_to_file=False, show_som_map=False, show_tsne_res=False):
        if show_som_map:
            pyb.figure(figsize=(7, 5))
            pyb.bone()
            pyb.pcolor(self.som.distance_map().T)
            pyb.colorbar()
            if save_image_to_file:
                pyb.savefig(get_file_path('map', overwrite=self.overwrite, directory=TARGET_DIR))
        if show_tsne_res:
            plt.figure(figsize=(20, 17))
            plt.scatter(self.tsne_results[:, 0],
                        self.tsne_results[:, 1],
                        s=15,
                        c='black',
                        edgecolors='none',
                        label='Clusters ')
            plt.title('Clusters')
            plt.xlabel('X')
            plt.ylabel('Y')
            if save_image_to_file:
                plt.savefig(get_file_path('map', overwrite=self.overwrite, directory=TARGET_DIR))
            plt.show()

        plt.figure(figsize=(16, 4))
        ax = sns.countplot(self.y)
        ax.set_title("Clusters sizes")
        for p in ax.patches:
            ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))
        if save_image_to_file:
            plt.savefig(get_file_path('count', overwrite=self.overwrite, directory=TARGET_DIR))
        plt.show()
        
        clusters_codes = pd.DataFrame(self.y, columns=['cl'])['cl'].unique()
        
        sizes = []
        labels = []
        colors = []

        cl_colors = np.random.choice(len(cm.get_cmap().colors), self.n_clusters)
        
        size_other = 0
        for i in range(0, self.n_clusters):
            cl_code = clusters_codes[i]
            if cl_code < 0:
                continue
            p_size = (len(self.y[self.y == cl_code]) * 100) / len(self.y)
            if p_size < 2.0:
                size_other += p_size
            else:
                sizes.append(p_size)
                labels.append(cl_code)
                colors.append(cm.get_cmap().colors[cl_colors[i]])

        if size_other > 0.0:
            sizes.append(size_other)
            labels.append('Other ( < 1%)')
            colors.append(cm.get_cmap().colors[cl_colors[len(sizes)]])
        
        pie_fig, pie_ax = plt.subplots(figsize=(16, 15))
        pie_ax.pie(sizes,
                   labels=labels, 
                   autopct='%1.1f%%', 
                   shadow=False, 
                   colors=colors,
                   startangle=90)
        pie_ax.axis('equal')
        if save_image_to_file:
            plt.savefig(get_file_path('pie', overwrite=self.overwrite, directory=TARGET_DIR))
        plt.show()
        
        plt.figure(figsize=(9, 7))

        for i in range(0, self.n_clusters):
            cl_code = clusters_codes[i]
            if cl_code < 0:
                continue
            plt.scatter(self.index_array[self.y == cl_code, 0], 
                        self.index_array[self.y == cl_code, 1], 
                        s=15,
                        c=cm.get_cmap().colors[cl_colors[i]],
                        edgecolors='none', 
                        label='Cluster ' + str(cl_code))
        plt.title('Clusters')
        plt.xlabel('X')
        plt.ylabel('Y')
        if save_image_to_file:
            plt.savefig(get_file_path('clusters', overwrite=self.overwrite, directory=TARGET_DIR))
        plt.show()

    def clusters_tsne(self):
        clusters_orig = []

        clusters_codes = pd.DataFrame(self.y, columns=['cl'])['cl'].unique()

        for n in range(0, self.n_clusters):
            cl_code = clusters_codes[n]
            if cl_code < 0:
                continue
            for i in range(0, len(self.y_pred)):
                if self.y_pred[i] == cl_code:
                    clusters_orig.append(cl_code)
                    clusters_orig.append(self.orig[i])
        return np.array(clusters_orig).reshape(int(len(clusters_orig) / 2), 2)

    def clusters(self):
        clusters_orig = []
        
        clusters_codes = pd.DataFrame(self.y, columns=['cl'])['cl'].unique()
        
        for n in range(0, self.n_clusters):
            cl_code = clusters_codes[n]
            if cl_code < 0:
                continue
            x_rows = self.get_original_x_rows_for_cluster(cl_code)
            for i in range(0, len(x_rows)):
                for j in range(0, len(self.X)):
                    if (x_rows[i] == self.X[j]).all() and self.orig[j] not in clusters_orig:
                        clusters_orig.append(cl_code)
                        clusters_orig.append(self.orig[j])
                        break
        return np.array(clusters_orig).reshape(int(len(clusters_orig)/2), 2)

    def get_clusters_rows_tsne(self):
        morph = MorphAnalyzer()

        clusters_orig = []
        clusters_list = []

        clusters_codes = pd.DataFrame(self.y, columns=['cl'])['cl'].unique()

        clusters = self.clusters_tsne()

        for n in range(0, self.n_clusters):
            cl_code = clusters_codes[n]
            if cl_code < 0:
                continue
            values = set()
            orig_values = set()
            for j in range(0, len(clusters)):
                if int(clusters[j][0]) == int(cl_code):
                    orig_values.add(clusters[j][1])
                    review = get_prepared_word_array(clusters[j][1])
                    filtered_review = []
                    for word in review:
                        if not word in self.stop_words_set:
                            parse_result = morph.parse(word)
                            if len(parse_result) == 1:
                                filtered_review.append(morph.parse(word)[0].normal_form)
                            elif len(parse_result) > 0:
                                normal_form = morph.parse(word)[0].normal_form
                                for k in range(0, len(parse_result)):
                                    if parse_result[k].tag.POS == 'NOUN':
                                        normal_form = morph.parse(word)[k].normal_form
                                        break
                                filtered_review.append(normal_form)
                            else:
                                filtered_review.append(word)
                    values.add(" ".join(np.array(filtered_review)))
            clusters_orig.append(orig_values)
            clusters_list.append(" ".join(list(values)))
        return clusters_orig, clusters_list

    def get_clusters_rows(self, limit=-1):
        morph = MorphAnalyzer()
        
        clusters_orig = []
        clusters_list = []
        
        clusters_codes = pd.DataFrame(self.y, columns=['cl'])['cl'].unique()
        
        for n in range(0, self.n_clusters):
            cl_code = clusters_codes[n]
            if cl_code < 0:
                continue
            x_rows = self.get_original_x_rows_for_cluster(cl_code, limit)
            values = set()
            orig_values = set()
            for i in range(0, len(x_rows)):
                for j in range(0, len(self.X)):
                    if (x_rows[i] == self.X[j]).all() and self.orig[j] not in orig_values:
                        orig_values.add(self.orig[j])
                        review = get_prepared_word_array(self.orig[j])                        
                        filtered_review = []
                        for word in review:
                            if not word in self.stop_words_set:
                                parse_result = morph.parse(word)
                                if len(parse_result) == 1:
                                    filtered_review.append(morph.parse(word)[0].normal_form)
                                elif len(parse_result) > 0:
                                    normal_form = morph.parse(word)[0].normal_form
                                    for k in range(0, len(parse_result)):
                                        if parse_result[k].tag.POS == 'NOUN':
                                            normal_form = morph.parse(word)[k].normal_form
                                            break
                                    filtered_review.append(normal_form)
                                else:
                                    filtered_review.append(word)
                        values.add(" ".join(np.array(filtered_review)))
                        break
            clusters_orig.append(orig_values)
            clusters_list.append(" ".join(list(values)))
        return clusters_orig, clusters_list

    def show_wordcloud_tsne(self, save_image_to_file=False):
        clusters_orig, clusters_list = self.get_clusters_rows_tsne()
        self.wordclouds(clusters_list=clusters_list, save_image_to_file=save_image_to_file)

    def show_wordcloud(self, save_image_to_file=False):
        clusters_orig, clusters_list = self.get_clusters_rows()
        self.wordclouds(clusters_list=clusters_list, save_image_to_file=save_image_to_file)
    
    def wordclouds(self, clusters_list, save_image_to_file=False, save_image_to_db=False):
        clusters_codes = pd.DataFrame(self.y, columns=['cl'])['cl'].unique()
        
        for i in range(0, len(clusters_list)):
            if len(clusters_list[i]) == 0:
                logger.info("Empty cluster #" + str(clusters_codes[i]))
                continue
            word_cloud = WordCloud(max_font_size=40, background_color="white").generate(clusters_list[i])
            plt.figure(figsize=(7, 4))
            plt.imshow(word_cloud, interpolation="bilinear")
            plt.axis("off")
            if save_image_to_file:
                plt.savefig(get_file_path('cl_' + str(clusters_codes[i]), directory=TARGET_DIR))
            
            if save_image_to_db:
                insert_cluster_logo(clusters_codes[i])
            
            plt.show()
    
    def set_additional_stopwords(self, add_stop_words):
        prev_length = len(self.stop_words_set)
        for i in range(0, len(add_stop_words)):
            self.stop_words_set.add(add_stop_words[i].strip())
        current_length = len(self.stop_words_set)
        logger.info("Added " + str(current_length - prev_length) + " stop words")
    
    def stopwords_from_file(self, filepath):
        file = open(os.path.abspath(filepath), "r")
        self.set_additional_stopwords(file.readlines())
    
    def get_stop_words_set(self):
        return self.stop_words_set
    
    def report(self, path, show_tsne_res=False):
        if show_tsne_res:
            clusters_orig, clusters_list = self.get_clusters_rows_tsne()
        else:
            clusters_orig, clusters_list = self.get_clusters_rows()

        self.wordclouds(clusters_list=clusters_list, save_image_to_file=True, save_image_to_db=True)
        
        clusters_orig_low, clusters_list_low = self.get_clusters_rows(10)
        
        pd.set_option('display.max_colwidth', -1)
        
        clusters_codes = pd.DataFrame(self.y, columns=['cl'])['cl'].unique()
        
        clusters_doc = []
        for i in range(0, len(clusters_list)):
            clusters_doc.append([
                    clusters_codes[i], 
                    os.path.abspath(TARGET_DIR + '/cl_' + str(clusters_codes[i]) + '.png'),
                    pd.DataFrame(list(clusters_orig_low[i])).to_html(header=False, classes=['greenTable']),
                    len(clusters_orig[i]),
                    self.min_msg_length])

        env = Environment(loader=FileSystemLoader(RESOURCE_DIR))
        template = env.get_template("template.html")
        template_vars = {
                "date": datetime.datetime.now().strftime("%d.%m.%Y"),
                "logo_img": os.path.abspath(TARGET_DIR + '/logo.png').replace("\\", "/"),
                "map_img": os.path.abspath(TARGET_DIR + '/map.png').replace("\\", "/"),
                "count_img": os.path.abspath(TARGET_DIR + '/count.png').replace("\\", "/"),
                "clusters_img": os.path.abspath(TARGET_DIR + '/clusters.png').replace("\\", "/"),
                "clusters": clusters_doc
                }
        html_out = template.render(template_vars)

        HTML(string=html_out).write_pdf(os.path.abspath(path))


class Classification:
    def __init__(self, max_features=250):
        self.max_features = max_features
        self.stop_words_set = set(stopwords.words('russian'))
    
    def set_additional_stopwords(self, add_stop_words):
        prev_length = len(self.stop_words_set)
        for i in range(0, len(add_stop_words)):
            self.stop_words_set.add(add_stop_words[i].strip())
        current_length = len(self.stop_words_set)
        logger.info("Added " + str(current_length - prev_length) + " stop words")
    
    def stopwords_from_file(self, filepath):
        file = open(os.path.abspath(filepath), "r")
        self.set_additional_stopwords(file.readlines())

    def build_classifier(self, hidden_layers, act_func, input_size, output_size):
        logger.info('==> build_classifier(' +
              str(hidden_layers) + ', ' +
              str(act_func) + ', ' +
              str(input_size) + ', ' +
              str(output_size) + ')')

        if input_size - output_size > 10:
            layer_size = input_size * 3
        else:
            layer_size = output_size * 7
        
        logger.info('Hidden layer size: ' + str(layer_size))

        inp = Input(shape=(input_size,))

        if hidden_layers > 0:
            nn = Dense(units=layer_size, kernel_initializer='he_normal', activation='relu')(inp)
            nn = Dropout(0.2)(nn)

            for i in range(1, hidden_layers):
                nn = Dense(units=layer_size, kernel_initializer='he_normal', activation='relu')(nn)
                nn = Dropout(0.2)(nn)

            outp = Dense(units=output_size, kernel_initializer='glorot_normal', activation='sigmoid')(nn)
        else:
            outp = Dense(units=output_size, kernel_initializer='glorot_normal', activation='sigmoid')(inp)

        model = Model(inputs=inp, outputs=outp)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        logger.info('<== build_classifier()')
        
        return model

    def fit(self, 
            dataset, 
            msg_column=1,
            class_column=0,
            min_msg_length=30,
            batch_size=10,
            nb_epoch=100,
            hidden_layers=2,
            act_func='relu', 
            save_image_to_file=False):
        
        corpus, orig = text_preprocessing(dataset,
                                          msg_column,
                                          min_msg_length,
                                          self.stop_words_set)
        
        logger.info('Complete text pre processing')

        cv = CountVectorizer(max_features=self.max_features)
        x = cv.fit_transform(corpus).toarray()
        y = [row[class_column] for row in dataset]
        y = np.array(y, dtype='int')

        logger.info('Complete determine X and y')

        lda = LDA(n_components=100)
        x = lda.fit_transform(x, y)

        logger.info('Complete Applying LDA. Found components: ' + str(len(x[0])))

        categories = pd.DataFrame(y)[0].unique()

        models = []
        valid_acc_scores = []
        splits = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(x, y))
        for idx, (train_idx, valid_idx) in enumerate(splits):
            logger.info('+-------------------------+')
            logger.info('| Fold: {:03d}               |'.format(idx + 1))
            logger.info('+-------------------------+')

            x_train = x[train_idx]
            y_train = y[train_idx]
            x_valid = x[valid_idx]
            y_valid = y[valid_idx]

            y_nn_train = self.y_label_to_onehot(y_train, categories)
            y_nn_valid = self.y_label_to_onehot(y_valid, categories)

            classifier = self.build_classifier(hidden_layers, act_func, len(x_train[0]), len(categories))
            classifier.fit(
                x_train,
                y_nn_train,
                callbacks=[
                    EarlyStopping(
                        monitor='val_loss',
                        mode='min',
                        restore_best_weights=True,
                        patience=5)
                ],
                batch_size=batch_size,
                epochs=nb_epoch,
                validation_data=(x_valid, y_nn_valid),
                verbose=2,
                shuffle=True)

            models.append(classifier)

            y_nn_pred = classifier.predict(x_valid)
            y_valid_pred = self.onehot_to_y_label(y_nn_pred, categories, 0.5)
            acc_score_val_fold = accuracy_score(y_valid, y_valid_pred)
            valid_acc_scores.append(acc_score_val_fold)

            logger.info('Accuracy score (threshold = 0.5): {:07.6f}'.format(round(acc_score_val_fold, 6)))

            plt.figure()
            plt.plot([0, 1], [0, 1], 'k--')

            for cl in range(len(categories)):
                cl_valid = [row[cl] for row in y_nn_valid]
                cl_pred = [row[cl] for row in y_nn_pred]
                fpr, tpr, _ = roc_curve(cl_valid, cl_pred)
                plt.plot(fpr, tpr)
                plt.legend(loc=4)

            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic. Fold ' + str(idx + 1))
            if save_image_to_file:
                plt.savefig(get_file_path('auc_fold' + str(idx + 1), directory=TARGET_DIR))
            plt.show()

        acc_score = np.asarray(valid_acc_scores, dtype='float64').mean()
        logger.info('Mean accuracy score (threshold = 0.5): {:07.6f}'.format(round(acc_score, 6)))

        with open(RESOURCE_DIR + '/complaint_classifier.pkl', 'wb') as file:
            pickle.dump((categories, cv, lda, models), file)

        logger.info('Dump classifier successfully')

        return acc_score

    def y_label_to_onehot(self, y, labels):
        y_nn = []
        for i in range(0, len(y)):
            for j in range(0, len(labels)):
                if labels[j] == y[i]:
                    y_nn.append(1)
                else:
                    y_nn.append(0)

        return np.array(y_nn, dtype='int').reshape(len(y), len(labels))

    def onehot_to_y_label(self, y_nn, labels, threshold):
        out = []
        for i in range(0, len(y_nn)):
            max_index = 0
            max_num = 0
            for j in range(0, len(labels)):
                if y_nn[i, j] > max_num:
                    max_num = y_nn[i, j]
                    max_index = j
            if y_nn[i, max_index] > threshold:
                out.append(labels[max_index])
            else:
                out.append(-1)

        return np.array(out, dtype='int')

    def predict_raw(self, messages):
        corpus, orig = text_preprocessing(messages, 0, 0, self.stop_words_set)
        
        with open(RESOURCE_DIR + '/complaint_classifier.pkl', 'rb') as fin:
            categories, cv, lda, models = pickle.load(fin)
        
        x = cv.transform(corpus).toarray()
        x = lda.transform(x)

        test_meta = np.zeros((x.shape[0], len(categories)))
        for model in models:
            test_meta += model.predict(x) / len(models)

        return categories, test_meta
    
    def predict(self, messages, threshold=0.5):
        categories, y_pred_nn = self.predict_raw(messages)
        
        logger.info('Using threshold: ' + str(threshold))

        return self.onehot_to_y_label(y_pred_nn, categories, threshold)
    
    def predict_backup(self, messages, date, clear_db=False, threshold=0.5):
        db_init()
        
        if clear_db:
            clear_classes()
            logger.info('All classes deleted')
        
        y_pred = self.predict(messages=messages, threshold=threshold)
        
        frame = pd.DataFrame(
            np.concatenate((y_pred.reshape(len(y_pred), 1), messages), axis=1), columns=['class', 'message']
        )
        
        unique_classes = frame['class'].unique()
        
        logger.info('Unique classes: ' + str(len(unique_classes)))
        
        for i in range(0, len(unique_classes)):
            class_num = int(unique_classes[i])
            class_rows = frame[frame['class'] == unique_classes[i]]
            cl_id = insert_class_row(class_num, len(class_rows), date)

            count = 10
            for cl in range(0, len(y_pred)):
                if y_pred[cl] == class_num:
                    if count > 0:
                        insert_msg_row(cl_id, class_num, str(messages[cl][0]))
                        count -= 1

            logger.info('class: ' + str(class_num) + ' \t\tsize: ' + str(len(class_rows)) + '. \t\tInsert to db successful.')

    def classes_report(self, path, msg_period=0):
        extract_all_logos()
        rows = get_all_classes()

        all_classes = np.array(([row[1] for row in rows]), dtype='int')
        all_ids = np.array(([row[0] for row in rows]), dtype='int')

        unique_classes = pd.DataFrame(all_classes, columns=['class'])['class'].unique()

        unique_classes.sort()

        all_sizes = np.array(([row[2] for row in rows]), dtype='int')
        all_dates = np.array(([row[3] for row in rows]))

        for i in range(0, len(unique_classes)):
            sizes = []
            dates = []
            for j in range(0, len(all_classes)):
                if unique_classes[i] == all_classes[j]:
                    sizes.append(all_sizes[j])
                    dates.append(all_dates[j])
            num_dates = []
            for k in range(0, len(dates)):
                num_dates.append(k)

            font_label = {'fontname': 'Arial', 'fontsize': '14'}
            font_title = {'fontname': 'Arial', 'fontsize': '18'}
            plt.figure(figsize=(10, 3))
            plt.bar(num_dates, sizes, width=1.0)
            plt.title(u'Класс ' + str(unique_classes[i]), **font_title)
            plt.xlabel(u'Дата', **font_label)
            plt.ylabel(u'Размер класса жалоб', **font_label)
            plt.xticks(
                [num_dates[0], num_dates[int(len(num_dates) / 2)], num_dates[len(num_dates) - 1]],
                [dates[0], dates[int(len(dates) / 2)], dates[len(dates) - 1]],
                rotation=45
            )
            plt.savefig(get_file_path('dyn_' + str(unique_classes[i]), directory=TARGET_DIR), bbox_inches='tight')
            plt.show()

        classes = []

        pd.set_option('display.max_colwidth', -1)

        reversed_all_classes = all_classes[::-1]
        reversed_all_ids = all_ids[::-1]
        for i in range(0, len(unique_classes)):
            messages = list()
            counter = 0
            for j in range(0, len(reversed_all_classes)):
                if unique_classes[i] == reversed_all_classes[j]:
                    msg_part = get_messages_by_class_id(int(reversed_all_ids[j]))
                    for msg in msg_part:
                        messages.append(msg)
                    if counter >= msg_period:
                        break
                    counter += 1

            classes.append([
                unique_classes[i],
                os.path.abspath(TARGET_DIR + '/cl_' + str(unique_classes[i]) + '.png'),
                os.path.abspath(TARGET_DIR + '/dyn_' + str(unique_classes[i]) + '.png'),
                pd.DataFrame(messages).to_html(header=False, classes=['greenTable'])
            ])

        env = Environment(loader=FileSystemLoader(RESOURCE_DIR))
        template = env.get_template("dynamics.html")
        template_vars = {
            "date": datetime.datetime.now().strftime("%d.%m.%Y"),
            "logo_img": os.path.abspath(TARGET_DIR + '/logo.png').replace("\\", "/"),
            "classes": classes
        }
        html_out = template.render(template_vars)

        HTML(string=html_out).write_pdf(os.path.abspath(path))
