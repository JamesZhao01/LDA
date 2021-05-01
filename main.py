import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
from gensim import corpora, models
from scipy.spatial import distance
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc


import nltk
import json
import os


def main():
    np.random.seed(2018)
    nltk.download('wordnet')
    stemmer = SnowballStemmer("english")

    def lemmatize_stemming(text):
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    def preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))
        return result

    def plot(matrix):
        plt.imshow(np.array(matrix))
        plt.show()
        # data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False)
        # data_text = data[['headline_text']]
        # data_text['index'] = data_text.index
        # documents = data_text

        # print(f"Documents length: {len(documents)}")
        # print("First five: ")
        # print(documents[:5])

        # processed_docs = documents['headline_text'][:10000].map(preprocess)
        # print(f"processed_docs: {processed_docs}")

        # with open("out.json", 'r') as infile:
        #     loaded_json = json.load(infile)
        #     loaded_json = [preprocess(i) for i in loaded_json]
        #     processed_docs = loaded_json

    files = []
    file_names = []
    dirs = ["BasicComputerSkills", "InternetSkills",
            "MicrosoftDigitalLiteracyCourse", "MicrosoftFiles"]
    dir_index = [0, 1, 2, 3]
    dirs = [dirs[i] for i in dir_index]

    for directory in dirs:
        path = f"./corpus/{directory}"
        for file in os.listdir(path):
            with open(os.path.join(path, file), "r", encoding='utf-8') as infile:
                files.append(infile.read())
                file_names.append(file)
    processed_docs = [preprocess(i) for i in files]

    for i in range(len(file_names)):
        print(f"i: {i}, file: {file_names[i]}")

    # Dictionary of counts
    dictionary = gensim.corpora.Dictionary(processed_docs)
    count = 0
    for k, v in dictionary.iteritems():
        print(k, v)
        count += 1
        if count > 10:
            break

    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    bow_doc_4310 = bow_corpus[0]
    # for i in range(len(bow_doc_4310)):
    #     print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], dictionary[bow_doc_4310[i][0]], bow_doc_4310[i][1]))

    # dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    # print(dictionary)

    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    # from pprint import pprint
    # for doc in corpus_tfidf:
    #     pprint(doc)
    #     break

    lda_model_tfidf = gensim.models.LdaMulticore(
        corpus_tfidf, num_topics=100, id2word=dictionary, passes=25, workers=4, minimum_probability=0)
    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))

    print("===TEST===")
    id = 0
    for index, score in sorted(lda_model_tfidf[bow_corpus[id]], key=lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(
            score, lda_model_tfidf.print_topic(index, 10)))

    # get themes on trained files
    results = [lda_model_tfidf[dictionary.doc2bow(
        pp_file)] for pp_file in processed_docs]
    results = [[i[1] for i in result] for result in results]

    np.save("./out/matrix", results)

    for i in range(len(results)):
        print(f"i: {i}, file: {file_names[i]}, themes: {results[i]}")

    print("===TEST_UNSEEN===")
    with open("./test/Python.txt", "r", encoding='utf-8') as infile:
        unseen_document = infile.read()
    preprocessed = preprocess(unseen_document)
    bow_vector = dictionary.doc2bow(preprocessed)
    print(lda_model_tfidf[bow_vector])
    for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
        print("Score: {}\t Topic: {}".format(
            score, lda_model_tfidf.print_topic(index, 5)))

    def pretty_print(matrix):
        for i in range(len(matrix)):
            print(["{:.3f}".format(round(item, 2)) for item in matrix[i]])

    print("=== J/S Distance Analysis ===")
    jensen_shannon = []
    for i in range(len(results)):
        jensen_shannon.append([0 for x in range(len(results))])
        for j in range(len(results)):
            jensen_shannon[i][j] = distance.jensenshannon(
                results[i], results[j])
    # pretty_print(jensen_shannon)

    print("=== Cos Distance Analysis ===")
    cos = []
    for i in range(len(results)):
        cos.append([0 for x in range(len(results))])
        for j in range(len(results)):
            cos[i][j] = distance.cosine(
                results[i], results[j])
    # pretty_print(cos)
    plot(cos)
    plt.figure(figsize=(10, 7))
    plt.title("Dendrograms")
    dend = shc.dendrogram(shc.linkage(jensen_shannon, method='ward'))
    plt.show()


if __name__ == "__main__":
    main()
