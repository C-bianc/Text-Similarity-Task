#!/usr/bin/env python
# coding: utf-8
#===============================================================================
#
#           FILE: 02_train_models.py 
#         AUTHOR: Bianca Ciobanica
#          EMAIL: bianca.ciobanica@student.uclouvain.be
#
#           BUGS: 
#        VERSION: 3.11.4
#        CREATED: 16-04-2024 
#
#===============================================================================
#    DESCRIPTION: Trains fasttext, word2vec and doc2vec models and saves them
#    
#   DEPENDENCIES: gensim, polars
#
#          USAGE: python 02_train_models.py 
#===============================================================================


from gensim.models import FastText, Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
import polars as pl
import multiprocessing
import os

cpus = multiprocessing.cpu_count()


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)


# load corpus
corpus = pl.read_ndjson("rtbf_subcorpus.ndjson")

# get tokenized and lemmatized 
texts = corpus.get_column("text_cleaned").to_list()
tagged_docs = [TaggedDocument(words=simple_preprocess(doc, deacc=False, min_len=2, max_len=25), tags=[i]) for i, doc in enumerate(texts)]


def create_model(texts, model_name, n_epochs=5, n_dimensions=100, n_min=1):
    # create model
    if (model_name.__name__ == "Doc2Vec"):
        model = model_name(workers=cpus-5, vector_size=n_dimensions, min_count=n_min, dm=1) # dm = distributed memory (not dbow) -> context is concatenated
    else:
        model = model_name(workers=cpus-5, vector_size=n_dimensions, min_count=n_min)
    
    # build vocab
    model.build_vocab(texts)
    
    # train 
    model.train(texts, total_examples=model.corpus_count, epochs=n_epochs)
         
    # save
    os.makedirs("models", exist_ok=True)
    model.save("models/{}_vsize{}.model".format(model_name.__name__, n_dimensions))

epochs = 25
dimensions=100 # change here

print("Creating Word2Vec model")
create_model(texts, Word2Vec, n_epochs=epochs, n_dimensions=dimensions, n_min=2)

print("Creating FastText model")
create_model(texts, FastText, n_epochs=epochs, n_dimensions=dimensions, n_min=2)

print("Creating Doc2Vec model")
create_model(tagged_docs, Doc2Vec, n_epochs=epochs, n_dimensions=dimensions, n_min=2)

print("Creating Doc2Vec model")
create_model(tagged_docs, Doc2Vec, n_epochs=25, n_dimensions=100, n_min=2)

