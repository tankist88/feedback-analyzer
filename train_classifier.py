import pickle

from ComplaintAnalizer import Classification

print('+-----------------------+')
print('| Let\'s go train!       |')
print('+-----------------------+')

with open('resources/clusters.pkl', 'rb') as file:
    clusters = pickle.load(file)

classifier = Classification()
classifier.stopwords_from_file('resources/complaint_stopwords.txt')
classifier.fit(dataset=clusters,
               batch_size=5,
               nb_epoch=50,
               min_msg_length=20,
               save_image_to_file=True)

print('Complete train')
