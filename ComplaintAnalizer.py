import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import pickle
import sqlite3


def get_prepared_word_array(text):
    line = re.sub('[^а-яА-Я]', ' ', text)
    line = line.lower()
    word_array = line.split()
    return word_array


def text_preprocessing(dataset, msg_column, min_msg_length, stop_words_set):
    from nltk.stem.snowball import SnowballStemmer
    from pandas import isnull
    
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
            if not word in stop_words_set:
                filtered_review.append(ss.stem(word))
        review = ' '.join(np.array(filtered_review))
        corpus.append(review)
        orig.append(text)
    return corpus, orig


def db_init():
    conn = sqlite3.connect('complaint_stat.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS COMPLAINT_CLASSES 
        (
            ID INTEGER PRIMARY KEY AUTOINCREMENT, 
            CL_NUM INTEGER, 
            CL_SIZE INTEGER, 
            CL_DATE TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS CLASSES_LOGO 
        (
            CL_NUM INTEGER PRIMARY KEY, 
            CL_LOGO BLOB
        )
    """)
    conn.commit()
    conn.close()


def clear_classes():
    conn = sqlite3.connect('complaint_stat.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM COMPLAINT_CLASSES")
    conn.commit()
    conn.close()
    print('All classes deleted')


def clear_clusters():
    conn = sqlite3.connect('complaint_stat.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM CLASSES_LOGO")
    conn.commit()
    conn.close()
    print('All clusters deleted')


class Clustering:
    def __init__(self, 
                 max_features=250,
                 som_threshold=0.5,
                 som_size=100,
                 som_sigma=1.0,
                 som_learning_rate=0.5,
                 msg_column=0,
                 min_msg_length=30,
                 period_start='____________',
                 period_end='____________',
                 overwrite=True):
        self.max_features = max_features
        self.som_threshold = som_threshold
        self.msg_column = msg_column
        self.min_msg_length = min_msg_length
        self.period_start = period_start
        self.period_end = period_end
        self.som_size = som_size
        self.som_sigma = som_sigma
        self.som_learning_rate = som_learning_rate
        self.overwrite = overwrite
        
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        
        self.stop_words_set = set(stopwords.words('russian'))

    def get_original_x_rows_for_cluster(self, cluster_num, limit=-1):
        if limit > 0:
            print('--- cluster ' + str(cluster_num) + '---')
        
        cluster_lines = []

        cluster = self.index_array[self.y == cluster_num]

        cluster = cluster[np.argsort(cluster[:, 2])]

        cluster_length = 0
        if limit > 0 and limit < len(cluster):
            cluster_length = limit
        else:
            cluster_length = len(cluster)

        for i in range(0, cluster_length):
            if limit > 0:
                print('cluster[' + str(i) + ']: ' + str(cluster[i][2]))
            array_of_arrays = self.mappings[(cluster[i][0], cluster[i][1])]
            for j in range(0, len(array_of_arrays)):
                cluster_lines.append(array_of_arrays[j])
        if len(cluster_lines) > 0:
            cluster_lines = self.sc.inverse_transform(np.array(cluster_lines))
        return cluster_lines
    
    def fit(self, dataset):
        self.corpus, self.orig = text_preprocessing(dataset, 
                                                    self.msg_column, 
                                                    self.min_msg_length, 
                                                    self.stop_words_set)
        
        print("corpus length: " + str(len(self.corpus)))
            
        fname = 'som_' + str(self.max_features) + '_' + str(self.som_size) + '_' + str(self.som_sigma).replace(".", "_") + '_' + str(self.som_learning_rate).replace(".", "_")
        
        if not self.overwrite:
            fname = self.free_file_name(fname, 'pkl')
        else:
            fname = fname + '.pkl'

        print("Using SOM model file: " + str(fname))
        
        if not os.path.isfile(fname):
            from sklearn.feature_extraction.text import CountVectorizer
            cv = CountVectorizer(max_features=self.max_features)
            self.X = cv.fit_transform(self.corpus).toarray()
            
            from sklearn.preprocessing import MinMaxScaler
            self.sc = MinMaxScaler(feature_range=(0, 1))
            self.X_scale = self.sc.fit_transform(self.X)
            
            from minisom import MiniSom
            self.som = MiniSom(x=self.som_size,
                               y=self.som_size,
                               input_len=self.max_features,
                               sigma=self.som_sigma,
                               learning_rate=self.som_learning_rate)
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
                if distance_map[i, j] < self.som_threshold:
                    indexes_coords.append(i)
                    indexes_coords.append(j)
                    indexes_dist.append(distance_map[i, j])
        
        coord_array = np.array(indexes_coords).reshape(int(len(indexes_coords)/2), 2)
        dist_array = np.array(indexes_dist).reshape(int(len(indexes_dist)), 1)
        
        from hdbscan import HDBSCAN
        hdbscan = HDBSCAN(min_cluster_size=5)
        y_pred = hdbscan.fit_predict(coord_array)
        
        next_cluster_num = self.get_next_cluster_num()
        
        self.y = []
        filtered_coords = []
        filtered_dists = []
        for i in range(0, len(y_pred)):
            if y_pred[i] < 0:
                continue
            filtered_coords.append(coord_array[i])
            filtered_dists.append(dist_array[i])
            self.y.append(int(y_pred[i]) + int(next_cluster_num))
        
        print('Next cluster num: ' + str(next_cluster_num))
        
        clusters_codes = pd.DataFrame(self.y, columns=['cl'])['cl'].unique()
        
        self.n_clusters = len(clusters_codes)
        
        coord_array = np.array(filtered_coords)
        dist_array = np.array(filtered_dists)
        
        self.index_array = np.concatenate((coord_array, dist_array), axis=1)
        
        print('Cluster codes (' + str(self.n_clusters) + '):')
        print(clusters_codes)
    
    def get_next_cluster_num(self):
        db_init()
        
        conn = sqlite3.connect('complaint_stat.db')
        
        cursor = conn.cursor()
        
        cursor.execute("SELECT CL_NUM FROM CLASSES_LOGO ORDER BY CL_NUM DESC")
        
        result = cursor.fetchone()
        
        conn.commit()
        
        conn.close()
        
        if pd.isnull(result):
            return int(0)
        else:
            return int(result[0]) + 1
    
    def som_mappings(self):
        self.mappings = self.som.win_map(self.X_scale)
    
    def get_clusters_number(self):
        return self.n_clusters
    
    def get_corpus(self):
        return self.corpus
    
    def score(self):
        dlist = []
        for i in range(0, len(self.index_array)):
            dlist.append(self.index_array[i, 2])
        return np.array(dlist, dtype='float64').mean()
    
    def free_file_name(self, fname, ext):
        new_fname = str(fname) + '.' + str(ext)
        index = 1
        while os.path.isfile(new_fname):
            new_fname = str(fname) + '_' + str(index) + '.' + str(ext)
            index = index + 1
        return new_fname
    
    def visualize(self, save_image_to_file=False):
        import pylab as pyb
        pyb.figure(figsize=(7, 5))
        pyb.bone()
        pyb.pcolor(self.som.distance_map().T)
        pyb.colorbar()
        if save_image_to_file:
            if not os.path.isdir(os.path.abspath('target')):
                os.mkdir(os.path.abspath('target'))
            fname = os.path.abspath('target/som.png')
            if not self.overwrite:
                fname = self.free_file_name(os.path.abspath('target/som'), 'png')
            pyb.savefig(fname)
        
        import seaborn as sns
        plt.figure(figsize=(7, 3))
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
                fname = self.free_file_name(os.path.abspath('target/count'), 'png')
            plt.savefig(fname)
        plt.show()
        
        available_colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'orange',
                            'yellow', 'brown', 'grey', 'navy', 'purple', 'lightcoral',
                            'lime', 'steelblue', 'indigo','olive', 'khaki', 'crimson',
                            'slateblue', 'gold', 'darkseagreen', 'violet', 'black']
        plt.figure(figsize=(7, 6))
        
        clusters_codes = pd.DataFrame(self.y, columns=['cl'])['cl'].unique()
        
        for i in range(0, self.n_clusters):
            cl_code = clusters_codes[i]
            if cl_code < 0:
                continue
            plt.scatter(self.index_array[self.y == cl_code, 0], 
                        self.index_array[self.y == cl_code, 1], 
                        s=25,
                        c=available_colors[i],
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
                fname = self.free_file_name(os.path.abspath('target/clusters'), 'png')
            plt.savefig(fname)
        plt.show()
    
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
        
    def get_clusters_rows(self, limit=-1):
        from pymorphy2 import MorphAnalyzer
        morph = MorphAnalyzer()
        
        clusters_orig = []
        clusters_list = []
        
        clusters_codes = pd.DataFrame(self.y, columns=['cl'])['cl'].unique()
        
        for n in range(0, self.n_clusters):
            cl_code = clusters_codes[n]
            if cl_code < 0:
                continue
            X_rows = self.get_original_x_rows_for_cluster(cl_code, limit)
            values = set()
            orig_values = set()
            for i in range(0, len(X_rows)):
                for j in range(0, len(self.X)):
                    if (X_rows[i] == self.X[j]).all() and self.orig[j] not in orig_values:
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
    
    def show_wordcloud(self, save_image_to_file=False):
        clusters_orig, clusters_list = self.get_clusters_rows()
        self.wordclouds(clusters_list=clusters_list, save_image_to_file=save_image_to_file)
    
    def wordclouds(self, clusters_list, save_image_to_file=False, save_image_to_db=False):
        from wordcloud import WordCloud
        
        clusters_codes = pd.DataFrame(self.y, columns=['cl'])['cl'].unique()
        
        for i in range(0, len(clusters_list)):
            if len(clusters_list[i]) == 0:
                print("Empty cluster #" + str(clusters_codes[i]))
                continue
            wordcloud = WordCloud(max_font_size=40, background_color="white").generate(clusters_list[i])
            plt.figure(figsize=(7,4))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            if save_image_to_file:
                if not os.path.isdir(os.path.abspath('target')):
                    os.mkdir(os.path.abspath('target'))
                fname = os.path.abspath('target/cl_' + str(clusters_codes[i]) + '.png')
                plt.savefig(fname)
            
            if save_image_to_db:
                self.insert_cluster_logo(clusters_codes[i])
            
            plt.show()
    
    def insert_cluster_logo(self, num):
        db_init()

        filepath = os.path.abspath('target/cl_' + str(num) + '.png')
        
        with open(filepath, 'rb') as f:
            ablob = f.read()
        
        conn = sqlite3.connect('complaint_stat.db')
        
        try:
            cursor = conn.cursor()
            
            cursor.execute("INSERT INTO CLASSES_LOGO VALUES(?, ?)", (int(num), sqlite3.Binary(ablob)))
            
            conn.commit()
        except Exception as e:
            print('Exception while insert logo to db')
        
        conn.close()
    
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
    
    def report(self, path):
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
        
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template("template.html")
        template_vars = {
                "date": datetime.datetime.now().strftime("%d.%m.%Y"),
                "logo_img": os.path.abspath('target/logo.png').replace("\\", "/"),
                "som_img": os.path.abspath('target/som.png').replace("\\", "/"),
                "count_img": os.path.abspath('target/count.png').replace("\\", "/"),
                "clusters_img": os.path.abspath('target/clusters.png').replace("\\", "/"),
                "clusters": clusters_doc
                }
        html_out = template.render(template_vars)
        
        from weasyprint import HTML
        HTML(string=html_out).write_pdf(os.path.abspath(path))


class Classification:
    def __init__(self, max_features=250):
        self.max_features = max_features
        
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        
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
        
    def get_confusion_matrix(self):
        return self.cm
    
    def fit(self, 
            dataset, 
            msg_column=1,
            class_column=0,
            min_msg_length=30,
            batch_size=10,
            nb_epoch=100,
            hidden_layers=2,
            act_func='relu'):
        
        corpus, orig = text_preprocessing(dataset,
                                          msg_column,
                                          min_msg_length,
                                          self.stop_words_set)
        
        print('Complete text preprocessing')
        
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features=self.max_features)
        X = cv.fit_transform(corpus).toarray()
        y = [row[class_column] for row in dataset]
        y = np.array(y, dtype='int')

        print('Complete determine X and y')

        # Splitting the dataset into the Training set and Test set
        from sklearn.cross_validation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
        
        print('Complete spliting into the Training set and Test set')
        
        # Applying LDA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA(n_components=50)
        X_train = lda.fit_transform(X_train, y_train)
        X_test = lda.transform(X_test)
        
        print('Complete Applying LDA. Found components: ' + str(len(X_train[0])))
        
        categories = pd.DataFrame(y)[0].unique()

        y_nn_train = []
        for i in range(0, len(y_train)):
            for j in range(0, len(categories)):
                if categories[j] == y_train[i]:
                    y_nn_train.append(1)
                else:
                    y_nn_train.append(0)
        
        y_nn_train = np.array(y_nn_train, dtype='int').reshape(len(y_train), len(categories))
        
        print('Complete create Y matrix')
        
        # Importing the Keras libraries and packages
        from keras.models import Sequential
        from keras.layers import Dense
        
        classifier = Sequential()
        classifier.add(Dense(output_dim=len(X_train[0]), init='uniform', activation=act_func, input_dim=len(X_train[0])))
        for i in range(0, hidden_layers):
            classifier.add(Dense(output_dim=len(X_train[0]), init='uniform', activation=act_func))
        classifier.add(Dense(output_dim=len(categories), init='uniform', activation='sigmoid'))
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        print('Complete create arch of NN and compile')
        
        # Fitting the ANN to the Training set
        classifier.fit(X_train, y_nn_train, batch_size=batch_size, nb_epoch=nb_epoch)
        
        print('Complete fitting NN')
        
        with open('complaint_classifier.pkl', 'wb') as fout:
            pickle.dump((categories, cv, lda, classifier), fout)
        
        # Predicting the Test set results
        y_pred_nn = classifier.predict(X_test)
        
        out = []
        for i in range(0, len(y_pred_nn)):
            max_index = 0
            max_num = 0
            for j in range(0, len(categories)):
                if y_pred_nn[i, j] > max_num:
                    max_num = y_pred_nn[i, j]
                    max_index = j
            out.append(categories[max_index])
        
        y_pred = np.array(out, dtype='int')
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        self.cm = confusion_matrix(y_test, y_pred)
        
        print('Confusion Matrix')
        print(self.cm)
    
    def predict_raw(self, messages):
        corpus, orig = text_preprocessing(messages, 0, 0, self.stop_words_set)
        
        with open('complaint_classifier.pkl', 'rb') as fin:
            categories, cv, lda, classifier = pickle.load(fin)
        
        x = cv.transform(corpus).toarray()
        x_test = lda.transform(x)
        
        # Predicting the Test set results
        return categories, classifier.predict(x_test)
    
    def predict(self, messages, threshold=0.5):
        # Predicting the Test set results
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
    
    def insert_class_row(self, num, size, date):
        conn = sqlite3.connect('complaint_stat.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO COMPLAINT_CLASSES VALUES(null, ?, ?, ?)", (num, size, date))
        conn.commit()
        conn.close()
        
    def show_all_classes(self):
        conn = sqlite3.connect('complaint_stat.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM COMPLAINT_CLASSES")
        results = cursor.fetchall()
        conn.close()
        return results
    
    def extract_all_logos(self):
        conn = sqlite3.connect('complaint_stat.db')
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
    
    def predict_backup(self, messages, date, clear_db=False, threshold=0.5):
        db_init()
        
        if clear_db:
            self.clear_classes()
            print('All classes deleted')
        
        y_pred = self.predict(messages=messages, threshold=threshold)
        
        frame = pd.DataFrame(np.concatenate((y_pred.reshape(len(y_pred), 1), messages), axis=1), columns=['class', 'message'])
        
        unique_classes = frame['class'].unique()
        
        print('Unique classes: ' + str(len(unique_classes)))
        
        for i in range(0, len(unique_classes)):
            class_num = int(unique_classes[i])
            class_rows = frame[frame['class'] == unique_classes[i]]
            self.insert_class_row(class_num, len(class_rows), date)
            print('class: ' + str(class_num) + ' \t\tsize: ' + str(len(class_rows)) + '. \t\tInsert to db successful.')
    
    def classes_report(self, path):
        self.extract_all_logos()
        rows = self.show_all_classes()

        all_classes = np.array(([row[1] for row in rows]), dtype='int')
        
        unique_classes = pd.DataFrame(all_classes, columns=['class'])['class'].unique()
        
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
            
            font_label = {'fontname': 'Arial', 'fontsize': '17'}
            font_title = {'fontname': 'Arial', 'fontsize': '20'}
            plt.figure(figsize=(10, 5))
            plt.bar(num_dates, sizes, width=1.0)
            plt.title(u'Кластер ' + str(unique_classes[i]), **font_title)
            plt.xlabel(u'Дата', **font_label)
            plt.ylabel(u'Размер класса жалоб', **font_label)
            plt.xticks(num_dates, dates, rotation=45)

            if not os.path.isdir(os.path.abspath('target')):
                os.mkdir(os.path.abspath('target'))
            filename = os.path.abspath('target/dyn_' + str(unique_classes[i]) + '.png')

            plt.savefig(filename, bbox_inches='tight')
            plt.show()
                
        classes = []

        for i in range(0, len(unique_classes)):
            classes.append([
                            unique_classes[i],
                            os.path.abspath('target/cl_' + str(unique_classes[i]) + '.png'),
                            os.path.abspath('target/dyn_' + str(unique_classes[i]) + '.png')
                            ])
        
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template("dynamics.html")
        template_vars = {
                "date": datetime.datetime.now().strftime("%d.%m.%Y"),
                "logo_img": os.path.abspath('target/logo.png').replace("\\", "/"),
                "classes": classes
                }
        html_out = template.render(template_vars)
        
        from weasyprint import HTML
        HTML(string=html_out).write_pdf(os.path.abspath(path))
