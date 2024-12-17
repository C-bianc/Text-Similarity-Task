#!/usr/bin/env python
# ~* coding: utf-8 *~
#===============================================================================
#
#           FILE: get_most_sim.py 
#         AUTHOR: Bianca Ciobanica
#          EMAIL: bianca.ciobanica@student.uclouvain.be
#
#           BUGS: 
#        VERSION: 3.11.4
#        CREATED: 27-05-2024 
#
#===============================================================================
#    DESCRIPTION: module for calculating similarities based on query 
#                 the module preprocesses the query, calculates cosine sim score,
#                 displays results and stores them
#    
#   DEPENDENCIES: polars, numpy, datetime, os, preprocessed_doc, random,
#                 string
#
#          USAGE: python get_most_sim.py 
#===============================================================================
import os
from datetime import datetime
import numpy as np
import polars as pl
from preprocess_doc import apply_preprocessing
import random
import string

# output dir for results
results_dir = "./resultats_requetes/"
os.makedirs(results_dir, exist_ok=True)


def find_document_by_id(doc_id, corpus, parser):
    doc = corpus.filter(pl.col("id") == doc_id) # filter by given id
    if len(doc) == 0: # no doc found
        raise ValueError ("Cet id est introuvable. Veuillez vous référencer à la liste des ids dans le fichier in all_ids.txt")

    tokens = doc.get_column("preprocessed_text").item()
    doc_text = doc.get_column("text_cleaned").item()

    return doc_text, tokens

def display_and_store_results(query_doc, sims, top_n, corpus, n_queries):

    n_best = np.argsort(sims)[::-1] # sort by best doc, n_best holds indices sorted in descending order

    # check if first to get real top n
    best_result = int(n_best[0])
    best_text = corpus[best_result].get_column("text_cleaned").item()
    if best_text == query_doc:
        n_best = n_best[1:top_n + 1] # ignore first
    else:
        n_best = n_best[:top_n]
    
    # store results
    results_df = pl.DataFrame()

    print("Query doc : ")
    print("_______________________________________")
    print(query_doc + "\n")

    for rank, i in enumerate(n_best, start=1):
        i = int(i)
        doc = corpus[i]
        doc_text = doc.get_column("text_cleaned").item()
        doc_score = sims[i]

        # concatenate results in one df
        doc = doc.with_columns(cosine_sim_score = pl.lit(doc_score)) # add query text
        doc = doc.with_columns(query = pl.lit(query_doc)) # add score
        doc = doc.drop('preprocessed_text') # remove unwanted column
        results_df = pl.concat([results_df, doc])
        
        # pretty print
        print(" ___________________________")
        print(f'|    [{rank}] Rank               |')
        print(f"|    Score : {doc_score:>8.2f}       |")
        print("|___________________________|\n")
        for metadata in corpus.columns:
            # ignore these features
            if metadata == "preprocessed_text" or metadata == "text_cleaned":
                continue
            info = corpus[i].get_column(metadata).item()
            if metadata == "pub_date":
                print("  pub date :", info.strftime("%d-%m-%y"), "(DD-MM-YYYY)")
            else:
                print("{:>10s} : {:>}".format(metadata, info))
    
        print("\nDocument text : \n")
        print(doc_text)
        print("_______________________________________")
        print()

    n_queries += 1
    suffix = ''.join(random.choices(string.ascii_letters, k=4))

    results_df.write_csv(os.path.join(results_dir, f'requete_{n_queries}_{suffix}.csv'))


def tokenize_query(args, parser, corpus):
    """ input: args (user input text)
        Preprocesses the user's query

        returns original text and tokenized text
    """

    # if doc id
    if args.i: 
        query_text, tokens = find_document_by_id(args.i, corpus, parser)
        return query_text, tokens

    # if text from user
    elif args.text: 
        # apply custom preprocessing fuc
        query_text = args.text
        tokens = apply_preprocessing(query_text)

        return query_text, tokens

def filter_corpus_by_date(args, corpus):
    filter_condition = pl.lit(True) 
    day = args.d
    month = args.m
    year = args.y

    # filter corpus if datetime filter in args
    if month:
        if day: # if month and day
            filter_condition &= ((pl.col("pub_date").dt.month() == month) & (pl.col("pub_date").dt.day() == day))

        elif year: # if month and year
            filter_condition &= ((pl.col("pub_date").dt.month() == month) & (pl.col("pub_date").dt.year() == year))

        # just month
        filter_condition &= pl.col("pub_date").dt.month() == month

    elif day:
        if year: # if day and year
            filter_condition &= ((pl.col("pub_date").dt.day() == day) & (pl.col("pub_date").dt.year() == year))
        # just day
        filter_condition &= pl.col("pub_date").dt.day() == day

    # just year
    elif year:
        filter_condition &= pl.col("pub_date").dt.year() == year


    filtered_corpus = corpus.filter(filter_condition) # filter & according to arguments
    indices = corpus.select(pl.arg_where(filter_condition)) # get indices of filtered rows

    return filtered_corpus, indices

def filter_corpus_by_date_range(args, corpus):
    filter_condition = pl.lit(True) 

    if args.start_date:
        if args.end_date:
            filter_condition &= pl.col("pub_date").is_between(args.start_date, args.end_date)
        else:
            filter_condition &= pl.col("pub_date") >= args.start_date

    filtered_corpus = corpus.filter(filter_condition) # filter & according to arguments
    indices = corpus.select(pl.arg_where(filter_condition)) # get indices of filtered rows

    return filtered_corpus, indices


#### MAIN ###
def process_query(args, parser, w2v_model, w2v_corpus_vectors, corpus):

    n_queries = 0

    # parse date in order to do query range
    corpus = corpus.with_columns(pl.col("pub_date").str.to_datetime("%Y-%m-%d %H:%M:%S%.f"))
    #print("before filter: ", len(corpus))


    # get tokens 
    query_text, tokens = tokenize_query(args, parser, corpus)
    
    ### FILTER CORPUS ACCORDING TO CHOSEN DATE ###

    # user defined a date and filter accordinly, but also vectors
    if args.y or args.m or args.d or args.start_date or args.end_date:
        if args.start_date or args.end_date:
            corpus, filtered_indices = filter_corpus_by_date_range(args, corpus)
        else:
            corpus, filtered_indices = filter_corpus_by_date(args, corpus)

        # filter vectors by the filtered indices for cosine similarity
        w2v_corpus_vectors = np.array(w2v_corpus_vectors)[np.array(filtered_indices.get_column("literal").to_list())]
        #print("after filter: ", len(filtered_documents))
        #print("filtered docs : ", len(filtered_indices))

    top_n = args.n

    if corpus.height < top_n and corpus.height != 0: # in case we have lower than 3 results in our filtered corpus, give best document
        top_n = 1
    elif top_n > corpus.height:
        parser.error(f'Erreur. Le nombre de documents similaires à afficher ne peut pas excéder la taille du corpus, à savoir {corpus.height}.', file=sys.stderr)
 
    ### Similarity ###
    # get mean vector of query with pre and post normalization
    query_vector = w2v_model.get_mean_vector(tokens, weights=None, pre_normalize=True, post_normalize=True, ignore_missing=True)

    # get cosine sims
    sims = w2v_model.cosine_similarities(query_vector, w2v_corpus_vectors)

    display_and_store_results(query_text, sims, top_n, corpus, n_queries)

