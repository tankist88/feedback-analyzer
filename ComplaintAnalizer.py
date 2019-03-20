import datetime
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
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from weasyprint import HTML
from wordcloud import WordCloud

nltk.download('stopwords')


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
    conn = sqlite3.connect('resources/complaint_stat.db')
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
    conn = sqlite3.connect('resources/complaint_stat.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM COMPLAINT_CLASSES")
    cursor.execute("DELETE FROM COMPLAINT_MESSAGES")
    conn.commit()
    conn.close()
    print('All classes deleted')


def clear_clusters():
    db_init()
    conn = sqlite3.connect('resources/complaint_stat.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM CLASSES_LOGO")
    conn.commit()
    conn.close()
    print('All clusters deleted')


def free_file_name(filename, ext):
    new_filename = str(filename) + '.' + str(ext)
    index = 1
    while os.path.isfile(new_filename):
        new_filename = str(filename) + '_' + str(index) + '.' + str(ext)
        index = index + 1
    return new_filename


def get_next_cluster_num():
    db_init()

    conn = sqlite3.connect('resources/complaint_stat.db')
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

    filepath = os.path.abspath('target/cl_' + str(num) + '.png')

    with open(filepath, 'rb') as f:
        ablob = f.read()

    conn = sqlite3.connect('resources/complaint_stat.db')

    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO CLASSES_LOGO VALUES(?, ?)", (int(num), sqlite3.Binary(ablob)))
        conn.commit()
    except Exception as e:
        print('Exception while insert logo to db')

    conn.close()


def insert_class_row(num, size, date):
    conn = sqlite3.connect('resources/complaint_stat.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO COMPLAINT_CLASSES VALUES(null, ?, ?, ?)", (num, size, date))
    conn.commit()
    cursor.execute("SELECT ID FROM COMPLAINT_CLASSES ORDER BY ID DESC")
    cl_id = cursor.fetchone()
    conn.close()
    return int(cl_id[0])


def insert_msg_row(cl_id, num, msg):
    conn = sqlite3.connect('resources/complaint_stat.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO COMPLAINT_MESSAGES VALUES(null, ?, ?, ?)", (cl_id, num, msg))
    conn.commit()
    conn.close()


def get_all_classes():
    conn = sqlite3.connect('resources/complaint_stat.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM COMPLAINT_CLASSES ORDER BY date(CL_DATE) ASC")
    results = cursor.fetchall()
    conn.close()
    return results


def get_messages_by_class_id(cl_id):
    conn = sqlite3.connect('resources/complaint_stat.db')
    cursor = conn.cursor()
    cursor.execute("SELECT CL_MSG FROM COMPLAINT_MESSAGES WHERE CL_ID=?", (cl_id,))
    results = cursor.fetchall()
    conn.close()
    return np.array(([row[0] for row in results]))


def extract_all_logos():
    conn = sqlite3.connect('resources/complaint_stat.db')
    cursor = conn.cursor()
    cursor.execute("SELECT CL_NUM, CL_LOGO FROM CLASSES_LOGO")
    results = cursor.fetchall()
    conn.close()

    if len(results) > 0:
        for result in results:
            if not os.path.isdir(os.path.abspath('target')):
                os.mkdir(os.path.abspath('target'))
            filename = os.path.abspath('target/cl_' + str(result[0]) + '.png')
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
            print('--- cluster ' + str(cluster_num) + '---')
        
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
                    print('cluster[' + str(i) + ']: ' + str(cluster[i][2]))
                array_of_arrays = self.mappings[(cluster[i][0], cluster[i][1])]
                for j in range(0, len(array_of_arrays)):
                    cluster_lines.append(array_of_arrays[j])
            if len(cluster_lines) > 0:
                cluster_lines = self.sc.inverse_transform(np.array(cluster_lines))
        else:
            for i in range(0, len(self.y_pred)):
                if self.y_pred[i] == cluster_num:
                    cluster_lines.append(self.X[i])
                if len(cluster_lines) >= cluster_length:
                    break
                
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
        
        print("corpus length: " + str(len(self.corpus)))
            
        fname = \
            'resources/som_' + \
            str(self.max_features) + '_' + \
            str(som_size) + '_' + \
            str(som_sigma).replace(".", "_") + '_' + \
            str(som_learning_rate).replace(".", "_")
        
        if not self.overwrite:
            fname = free_file_name(fname, 'pkl')
        else:
            fname = fname + '.pkl'

        print("Using SOM model file: " + str(fname))
        
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
        
        print('X rows: ' + str(len(self.X)))
        print('X cols: ' + str(len(self.X[0])))
        
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
        
        print('Next cluster num: ' + str(next_cluster_num))
        
        clusters_codes = pd.DataFrame(self.y, columns=['cl'])['cl'].unique()
        
        self.n_clusters = len(clusters_codes)
        
        coord_array = np.array(filtered_coords)
        dist_array = np.array(filtered_dists)
        
        self.index_array = np.concatenate((coord_array, dist_array), axis=1)
        
        print('Cluster codes (' + str(self.n_clusters) + '):')
        print(clusters_codes)
        
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
        
        print("corpus length: " + str(len(self.corpus)))
            
        fname = \
            'resources/tsne_' + \
            str(self.max_features) + '_' + \
            str(n_iter) + '_' +  \
            str(n_components) + '_' + \
            str(perplexity).replace(".", "_") + '_' + \
            str(learning_rate).replace(".", "_")
        
        if not self.overwrite:
            fname = free_file_name(fname, 'pkl')
        else:
            fname = fname + '.pkl'

        print("Using t-SNE model file: " + str(fname))
        
        if not os.path.isfile(fname):
            cv = CountVectorizer(max_features=self.max_features)
            self.X = cv.fit_transform(self.corpus).toarray()

            pca = PCA(n_components=n_components)
            x_pca = pca.fit_transform(self.X)
            print('Cumulative explained variation for principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))

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
        
        print('X rows: ' + str(len(self.X)))
        print('X cols: ' + str(len(self.X[0])))
        
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
        
        print('Next cluster num: ' + str(next_cluster_num))
        
        clusters_codes = pd.DataFrame(self.y, columns=['cl'])['cl'].unique()
        
        self.n_clusters = len(clusters_codes)
        
        filtered_values = np.array(filtered_values)
        
        self.index_array = np.concatenate((filtered_values, np.zeros((len(filtered_values), 1), dtype='float')), axis=1)
        
        print('Cluster codes (' + str(self.n_clusters) + '):')
        print(clusters_codes)
    
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
                if not os.path.isdir(os.path.abspath('target')):
                    os.mkdir(os.path.abspath('target'))
                fname = os.path.abspath('target/map.png')
                if not self.overwrite:
                    fname = free_file_name(os.path.abspath('target/map'), 'png')
                pyb.savefig(fname)
        if show_tsne_res:
            plt.figure(figsize=(20, 18))
            plt.scatter(self.tsne_results[:, 0],
                        self.tsne_results[:, 1],
                        s=25,
                        c='black',
                        edgecolors='none',
                        label='Clusters ')
            plt.title('Clusters')
            plt.xlabel('X')
            plt.ylabel('Y')
            if save_image_to_file:
                if not os.path.isdir(os.path.abspath('target')):
                    os.mkdir(os.path.abspath('target'))
                fname = os.path.abspath('target/map.png')
                if not self.overwrite:
                    fname = free_file_name(os.path.abspath('target/map'), 'png')
                plt.savefig(fname)
            plt.show()

        plt.figure(figsize=(14, 3))
        ax = sns.countplot(self.y)
        ax.set_title("Clusters sizes")
        for p in ax.patches:
            ax.annotate(str(format(int(p.get_height()), ',d')), 
                        (p.get_x(), p.get_height()*1.01))
        if save_image_to_file:
            if not os.path.isdir(os.path.abspath('target')):
                os.mkdir(os.path.abspath('target'))
            fname = os.path.abspath('target/count.png')
            if not self.overwrite:
                fname = free_file_name(os.path.abspath('target/count'), 'png')
            plt.savefig(fname)
        plt.show()
        
        plt.figure(figsize=(7, 6))
        
        clusters_codes = pd.DataFrame(self.y, columns=['cl'])['cl'].unique()

        cl_colors = np.random.choice(len(cm.get_cmap().colors), self.n_clusters)

        for i in range(0, self.n_clusters):
            cl_code = clusters_codes[i]
            if cl_code < 0:
                continue
            plt.scatter(self.index_array[self.y == cl_code, 0], 
                        self.index_array[self.y == cl_code, 1], 
                        s=25,
                        c=cm.get_cmap().colors[cl_colors[i]],
                        edgecolors='none', 
                        label='Cluster ' + str(cl_code))
        plt.title('Clusters')
        plt.xlabel('X')
        plt.ylabel('Y')
        if save_image_to_file:
            if not os.path.isdir(os.path.abspath('target')):
                os.mkdir(os.path.abspath('target'))
            fname = os.path.abspath('target/clusters.png')
            if not self.overwrite:
                fname = free_file_name(os.path.abspath('target/clusters'), 'png')
            plt.savefig(fname)
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
                print("Empty cluster #" + str(clusters_codes[i]))
                continue
            word_cloud = WordCloud(max_font_size=40, background_color="white").generate(clusters_list[i])
            plt.figure(figsize=(7, 4))
            plt.imshow(word_cloud, interpolation="bilinear")
            plt.axis("off")
            if save_image_to_file:
                if not os.path.isdir(os.path.abspath('target')):
                    os.mkdir(os.path.abspath('target'))
                fname = os.path.abspath('target/cl_' + str(clusters_codes[i]) + '.png')
                plt.savefig(fname)
            
            if save_image_to_db:
                insert_cluster_logo(clusters_codes[i])
            
            plt.show()
    
    def set_additional_stopwords(self, add_stop_words):
        prev_length = len(self.stop_words_set)
        for i in range(0, len(add_stop_words)):
            self.stop_words_set.add(add_stop_words[i].strip())
        current_length = len(self.stop_words_set)
        print("Added " + str(current_length - prev_length) + " stop words")
    
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
        
        clusters_orig_low, clusters_list_low = self.get_clusters_rows(7)
        
        pd.set_option('display.max_colwidth', -1)
        
        clusters_codes = pd.DataFrame(self.y, columns=['cl'])['cl'].unique()
        
        clusters_doc = []
        for i in range(0, len(clusters_list)):
            clusters_doc.append([
                    clusters_codes[i], 
                    os.path.abspath('target/cl_' + str(clusters_codes[i]) + '.png'),
                    pd.DataFrame(list(clusters_orig_low[i])).to_html(
                            header=False, classes=['greenTable']),
                    len(clusters_orig[i]),
                    self.min_msg_length])

        env = Environment(loader=FileSystemLoader('resources'))
        template = env.get_template("template.html")
        template_vars = {
                "date": datetime.datetime.now().strftime("%d.%m.%Y"),
                "logo_img": os.path.abspath('target/logo.png').replace("\\", "/"),
                "map_img": os.path.abspath('target/map.png').replace("\\", "/"),
                "count_img": os.path.abspath('target/count.png').replace("\\", "/"),
                "clusters_img": os.path.abspath('target/clusters.png').replace("\\", "/"),
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
        print("Added " + str(current_length - prev_length) + " stop words")
    
    def stopwords_from_file(self, filepath):
        file = open(os.path.abspath(filepath), "r")
        self.set_additional_stopwords(file.readlines())

    def build_classifier(self, hidden_layers, act_func, input_size, output_size):
        print('==> build_classifier(' + str(hidden_layers) + ', ' + str(act_func) + ', ' + str(input_size) + ', ' + str(output_size) + ')')

        if input_size - output_size > 10:
            layer_size = input_size * 3
        else:
            layer_size = output_size * 7
        
        print('Hidden layer size: ' + str(layer_size))
        
        classifier = Sequential()
        classifier.add(Dense(output_dim=layer_size, init='uniform', activation=act_func, input_dim=input_size))
        for i in range(0, hidden_layers):
            classifier.add(Dense(output_dim=layer_size, init='uniform', activation=act_func))
            classifier.add(Dropout(0.2))
        classifier.add(Dense(output_dim=output_size, init='uniform', activation='sigmoid'))
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        print('<== build_classifier()')
        
        return classifier

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
        
        print('Complete text pre processing')

        cv = CountVectorizer(max_features=self.max_features)
        x = cv.fit_transform(corpus).toarray()
        y = [row[class_column] for row in dataset]
        y = np.array(y, dtype='int')

        print('Complete determine X and y')

        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1, random_state=0)
        
        print('Complete splitting into the Training set and Test set')

        lda = LDA(n_components=100)
        x_train = lda.fit_transform(x_train, y_train)
        x_valid = lda.transform(x_valid)
        
        print('Complete Applying LDA. Found components: ' + str(len(x_train[0])))
        
        categories = pd.DataFrame(y)[0].unique()

        y_nn_train = []
        for i in range(0, len(y_train)):
            for j in range(0, len(categories)):
                if categories[j] == y_train[i]:
                    y_nn_train.append(1)
                else:
                    y_nn_train.append(0)
        
        y_nn_train = np.array(y_nn_train, dtype='int').reshape(len(y_train), len(categories))

        y_nn_valid = []
        for i in range(0, len(y_valid)):
            for j in range(0, len(categories)):
                if categories[j] == y_valid[i]:
                    y_nn_valid.append(1)
                else:
                    y_nn_valid.append(0)

        y_nn_valid = np.array(y_nn_valid, dtype='int').reshape(len(y_valid), len(categories))
        
        print('Complete create Y matrix')
        classifier = self.build_classifier(hidden_layers, act_func, len(x_train[0]), len(categories))
        print('Complete create arch of NN and compile')
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
        print('Complete fitting NN')

        with open('resources/complaint_classifier.pkl', 'wb') as file:
            pickle.dump((categories, cv, lda, classifier), file)

        print('Dump classifier successfully')

    def predict_raw(self, messages):
        corpus, orig = text_preprocessing(messages, 0, 0, self.stop_words_set)
        
        with open('resources/complaint_classifier.pkl', 'rb') as fin:
            categories, cv, lda, classifier = pickle.load(fin)
        
        x = cv.transform(corpus).toarray()
        x_test = lda.transform(x)

        return categories, classifier.predict(x_test)
    
    def predict(self, messages, threshold=0.5):
        categories, y_pred_nn = self.predict_raw(messages)
        
        print('Using threshold: ' + str(threshold))
        
        out = []
        for i in range(0, len(y_pred_nn)):
            max_index = 0
            max_num = 0
            for j in range(0, len(categories)):
                if y_pred_nn[i, j] > max_num:
                    max_num = y_pred_nn[i, j]
                    max_index = j
            if y_pred_nn[i, max_index] > threshold:
                out.append(categories[max_index])
            else:
                out.append(-1)
        
        y_pred = np.array(out, dtype='int')
        
        return y_pred
    
    def predict_backup(self, messages, date, clear_db=False, threshold=0.5):
        db_init()
        
        if clear_db:
            clear_classes()
            print('All classes deleted')
        
        y_pred = self.predict(messages=messages, threshold=threshold)
        
        frame = pd.DataFrame(
            np.concatenate((y_pred.reshape(len(y_pred), 1), messages), axis=1), columns=['class', 'message']
        )
        
        unique_classes = frame['class'].unique()
        
        print('Unique classes: ' + str(len(unique_classes)))
        
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

            print('class: ' + str(class_num) + ' \t\tsize: ' + str(len(class_rows)) + '. \t\tInsert to db successful.')

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

            if not os.path.isdir(os.path.abspath('target')):
                os.mkdir(os.path.abspath('target'))
            filename = os.path.abspath('target/dyn_' + str(unique_classes[i]) + '.png')

            plt.savefig(filename, bbox_inches='tight')
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
                os.path.abspath('target/cl_' + str(unique_classes[i]) + '.png'),
                os.path.abspath('target/dyn_' + str(unique_classes[i]) + '.png'),
                pd.DataFrame(messages).to_html(header=False, classes=['greenTable'])
            ])

        env = Environment(loader=FileSystemLoader('resources'))
        template = env.get_template("dynamics.html")
        template_vars = {
            "date": datetime.datetime.now().strftime("%d.%m.%Y"),
            "logo_img": os.path.abspath('target/logo.png').replace("\\", "/"),
            "classes": classes
        }
        html_out = template.render(template_vars)

        HTML(string=html_out).write_pdf(os.path.abspath(path))
