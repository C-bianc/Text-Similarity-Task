#!/usr/bin/env python
# coding: utf-8
#===============================================================================
#
#           FILE: 03_evaluate_models.py 
#         AUTHOR: Bianca Ciobanica
#          EMAIL: bianca.ciobanica@student.uclouvain.be
#
#           BUGS: 
#        VERSION: 3.11.4
#        CREATED: 16-04-2024 
#
#===============================================================================
#    DESCRIPTION: evaluates model on the eval set and stores the results
#    
#   DEPENDENCIES: gensim, matplotlib, numpy, sklearn, scipy, polars
#
#          USAGE: python 03_evaluate_models.py 
#===============================================================================

from gensim.models.word2vec import Word2Vec
from gensim.models import FastText, Doc2Vec
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
from collections import defaultdict
from gensim import matutils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from scipy.linalg import orthogonal_procrustes
import re
import csv
import polars as pl
import multiprocessing

cpus = multiprocessing.cpu_count()

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)



corpus = pl.read_ndjson("rtbf_subcorpus.ndjson")
eval_set = pl.read_ndjson("evaluation_sets/evaluation_50_samples_tokenized.ndjson")



def get_keyed_vectors(vsize):
    w2v_model = Word2Vec.load(f'models/Word2Vec_vsize{vsize}.model', mmap=None)
    fasttext_model = FastText.load(f'models/FastText_vsize{vsize}.model',mmap=None)
    
    w2v_wv = w2v_model.wv
    fasttext_wv = fasttext_model.wv
    
    w2v_wv.save(f'models/w2v_vectors_v{vsize}.kv')
    fasttext_wv.save(f'models/fasttext_vectors_v{vsize}.kv') 
#get_keyed_vectors(100)
#get_keyed_vectors(200)

#w2v_model = KeyedVectors.load("models/w2v_vectors_v100.kv", mmap=None)
#fasttext_model = KeyedVectors.load("models/fasttext_vectors_v100.kv",mmap=None)

w2v_model = KeyedVectors.load("models/w2v_vectors_v200.kv", mmap=None)
fasttext_model = KeyedVectors.load("models/fasttext_vectors_v200.kv",mmap=None)


#doc2v_model = Doc2Vec.load("models/Doc2Vec_vsize100.model", mmap=None)


text_documents = corpus.get_column("preprocessed_text").to_list()


w2v_dimensions = w2v_model.vector_size
fasttext_dimensions = fasttext_model.vector_size
w2v_nb_words = len(w2v_model.index_to_key)
fasttext_nb_words = len(fasttext_model.index_to_key)

print("Features in W2V : ", w2v_dimensions)
print("Total words in W2V :", w2v_nb_words)
print("Total words in Fasttext :", fasttext_nb_words)



def get_most_sim(model, words, n):
    words = list(map(str.lower, words))
    return model.most_similar(positive=words, topn=n)

def get_most_distant(model, words):
    words = list(map(str.lower, words))
    return model.doesnt_match(words)
    
def get_sim_and_without(model, positive_words, negative_words):
    positive_words = list(map(str.lower, positive_words))
    negative_words = list(map(str.lower, negative_words))
    return model.most_similar(positive=positive_words, negative=negative_words)

def get_closer_than(model, a, from_a):
    return model.closer_than(a.lower(), from_a.lower())



query = "covid"
print("w2v results for : " + query)
print(get_most_sim(w2v_model, [query], 10))

print("fasttext results for : " + query)
print(get_most_sim(fasttext_model, [query], 10))


get_most_distant(w2v_model, ['waimes','malmedy','carnaval','croustillon','lac'])


get_sim_and_without(w2v_model, positive_words=['bruxelles','institution','europe'], negative_words=['liège','namur','guerre'])


get_sim_and_without(w2v_model, positive_words=['sncb','retard','travail'], negative_words=['vélo','stib','tec'])


def tsne_visualization(words, embedding_model, tsne_model):
    """
    dimensionalty reduction of word embeddings using tsne model

    input:
        list of words, tsne model

    returns:
        tuple of x,y coordinates and labels of each word
    """
    labels = []; tokens = []
    for word in words:
        tokens.append(embedding_model.wv[word])
        labels.append(word)

    tokens = np.array(tokens)
    values = tsne_model.fit_transform(tokens)

    return values, labels


def plot_scatter_with_annotations(values, labels):
    """
    creates 2D plot from word embeddings

    input:
        tuple of x,y coords and labels of each word
    
    """
    
    x = []; y = []
    for value in values:
        x.append(value[0])
        y.append(value[1])
        
    # plot params
    plt.rcParams.update({'font.size': 22, 'font.family': 'sans-serif', 'font.weight': 'normal'})
    plt.figure(figsize=(30, 18))
    plt.tight_layout()

    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i], 
                     xy=(x[i], y[i]), 
                     xytext=(10, 0), 
                     textcoords='offset points', 
                     ha='left', 
                     va='center')
    
    plt.show()


words = "bière fromage malmedy waimes sncb retard carrière plongée informatique" + \
" liège covid virtuel jeux sport train femme virus voiture europe bruxelles chocolat"
words = words.split()



# create tsne model
tsne_model = TSNE(perplexity=5, n_components=2, init='pca', n_iter=10000, random_state=39)
values, labels = tsne_visualization(words, w2v_model, tsne_model) 

plot_scatter_with_annotations(values, labels) # draw plot


simple_w2v_corpus_vectors = list(map(lambda doc: w2v_model.get_mean_vector(doc, weights=None, pre_normalize=True, post_normalize=True, ignore_missing=True), text_documents))


improved_w2v_corpus_vectors = list(map(lambda doc : get_pooled_vector(doc), text_documents))


def get_oov_vector(word):
    oov_vector = fasttext_model[word]
    transformed_vector = np.dot(oov_vector, rotation) # map fasttext vector to w2v space
    
    return transformed_vector



def get_pooled_vector(doc):
    vectors = []
    for word in doc:
        try:
            vector = w2v_model[word]
        except KeyError:
            vector = get_oov_vector(word)
        vectors.append(vector)
        
    mean_vector = np.mean(vectors, axis=0)
    max_vector = np.max(vectors, axis=0)
    
    return np.concatenate([mean_vector, max_vector])


def infer_doc2v(query_doc):
    inferred_vector = doc2v_model.infer_vector(query_doc)
    similar_docs = doc2v_model.dv.most_similar(inferred_vector, topn=3)
    
    results = []

    for doc_idx, score in similar_docs:
        text = corpus[int(doc_idx)].get_column("text_cleaned").to_list()
        text = "TEXT : \n" + text[0] + "\n\n SCORE : \n" + str(round(score, 2))
        results.append(text)
    
    return results


def infer_simple_w2v(query_doc):
    query_doc_vector = w2v_model.get_mean_vector(query_doc, weights=None, pre_normalize=True, post_normalize=True, ignore_missing=True)
    sims = w2v_model.cosine_similarities(query_doc_vector, simple_w2v_corpus_vectors)
    ranked_indices = np.argsort(sims)[::-1][:3]  # get idx sorted by similarity (descending order)
    similar_docs = [(idx, sims[idx]) for idx in ranked_indices]

    results = []

    for doc_idx, score in similar_docs:
        text = corpus[int(doc_idx)].get_column("text_cleaned").to_list()
        text = "TEXT : \n" + text[0] + "\n\n SCORE : \n" + str(round(score, 2))
        results.append(text)

    return results


def infer_improved_w2v(query_doc):
    query_doc_vector = get_pooled_vector(query_doc)
    sims = w2v_model.cosine_similarities(query_doc_vector, improved_w2v_corpus_vectors)
    ranked_indices = np.argsort(sims)[::-1][:3]  # get idx sorted by similarity (descending order)
    similar_docs = [(idx, sims[idx]) for idx in ranked_indices]

    results = []

    for doc_idx, score in similar_docs:
        text = corpus[int(doc_idx)].get_column("text_cleaned").to_list()
        text = "TEXT : \n" + text[0] + "\n\n SCORE : \n" + str(round(score, 2))
        results.append(text)

    return results


def infer_all_models(eval_set):
    results = []
    
    for i in range(eval_set.height):
        doc = eval_set[i]
        doc_id = doc.select("id").item()
        tokens = doc.select("preprocessed_text").item().to_list()
        doc_text = doc.select("text_cleaned").item()
        
        simple_w2v_results = infer_simple_w2v(tokens)
        improved_w2v_results = infer_improved_w2v(tokens)
        doc2v_results = infer_doc2v(tokens)

        for j in range(3):
            row = {
                "id": int(doc_id),
                "query_tokens": " ".join(tokens),
                "query_doc": doc_text,
                "rank": j + 1,
                "simple_w2v_results": simple_w2v_results[j],
                "improved_w2v_results": improved_w2v_results[j],
                "doc2v_results": doc2v_results[j]
            }
            results.append(row)
    return results
    
eval_df_results = infer_all_models(eval_set)


def write_results(results):
    with open('evaluation_set_v100.csv', 'w', newline='') as csvfile:
        fieldnames = ['id', 'query_tokens', 'query_doc', 'rank', 'simple_w2v_results', 'improved_w2v_results', 'BEST' ,'is_self']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
        writer.writeheader()
    
        for row in results:
            writer.writerow(row)
#write_results(eval_df_results)

