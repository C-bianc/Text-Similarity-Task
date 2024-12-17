#!/usr/bin/env python
# coding: utf-8
#===============================================================================
#
#           FILE: store_corpus_vectors.py 
#         AUTHOR: Bianca Ciobanica
#          EMAIL: bianca.ciobanica@student.uclouvain.be
#
#           BUGS: 
#        VERSION: 3.11.4
#        CREATED: 16-04-2024 
#
#===============================================================================
#    DESCRIPTION:  stores the list of mean vectors from whole coprus
#    
#   DEPENDENCIES: gensim, polars
#
#          usage: python store_corpus_vectors.py 
#===============================================================================


from gensim.models import KeyedVectors
import polars as pl
import pickle


w2v_model = KeyedVectors.load("model/w2v_vectors_v200.kv", mmap=None)
corpus = pl.read_ndjson("rtbf_subcorpus.ndjson")

#  get texts
text_documents = corpus.get_column("preprocessed_text").to_list()

# get all vectors
w2v_corpus_vectors = list(map(lambda doc: w2v_model.get_mean_vector(doc, weights=None, pre_normalize=True, post_normalize=True, ignore_missing=True), text_documents))

# store
with open('final_w2v_corpus_vectors.pkl', 'wb') as f:
    pickle.dump(w2v_corpus_vectors, f)

