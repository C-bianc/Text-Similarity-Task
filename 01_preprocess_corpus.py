#!/usr/bin/env python
# coding: utf-8
#===============================================================================
#
#           FILE: 01_preprocess_corpus.py 
#         AUTHOR: Bianca Ciobanica
#          EMAIL: bianca.ciobanica@student.uclouvain.be
#
#           BUGS: 
#        VERSION: 3.11.4
#        CREATED: 22-03-2024 
#
#===============================================================================
#    DESCRIPTION:  Filters the corpus and creates a subcorpus and evaluation set
#                  then tokenizes and lemmatizes the subcorpus
#                  Change the rtbf file path accordingly
#    
#   DEPENDENCIES: spacy, polars 
#
#          USAGE: python 01_preprocess_corpus.py
#===============================================================================

import sys
import os
# from gensim.utils import simple_preprocess
import spacy
import polars as pl
import re


rtbf_file = "../pre_processing/rtbfCorpus.json" # corpus path here

file_root, file_extension = os.path.splitext(rtbf_file)

# get file name
filename = os.path.basename(file_root)


steps = 1
def print_head(title):
    global steps 
    
    print("\n\n", str(steps) + ") " + title)
    print("____________________________________________________\n")
    steps += 1


select_range_query = (
    pl.scan_ndjson(rtbf_file)
    .with_columns(pl.col("pub_date").str.to_datetime("%Y-%m-%d %H:%M:%S%.f")) # parse date
    .with_columns(pl.col("pub_date").dt.year().alias("year")) # add new column year
    .filter((pl.col("year") == 2021) | (pl.col("year") == 2020)) # select by year
    .filter(pl.col("feed") == "RTBFINFO") # select feed RTBFINFO
)


select_range_query_eval = (
    pl.scan_ndjson(rtbf_file)
    .with_columns(pl.col("pub_date").str.to_datetime("%Y-%m-%d %H:%M:%S%.f")) # parse date
    .with_columns(pl.col("pub_date").dt.year().alias("year")) # add new column year
    .filter((pl.col("year") < 2020)) # select by year
    .filter(pl.col("feed") == "RTBFINFO") # select feed RTBFINFO
)


try:
    # load corpus
    print_head(f'Loading data from {filename}')
    
    file = '../pre_processing/rtbfCorpus.json'
    
    print('Preparing training and evaluation set...\n' + select_range_query.explain() + '\n')
    data = select_range_query.collect().unique() # collect data
    data = data.drop(["text_html", "test_preprocessed"]) # remove unwanted cols

    unseen_data = select_range_query_eval.collect().unique()
    unseen_data = unseen_data.drop(["text_html", "test_preprocessed"])
    
    evaluation_sample = pl.concat([data.sample(25), unseen_data.sample(25)])
    
    print(" \u2713 Data loaded and evaluation set created successfully!\n")
    
except Exception as e:
    print(f'Error : {e}.\n Please provide the correct rtbfCorpus.')


print("Total number of articles : ", data.height)
print("Columns : ", list(data.schema.keys()))



print_head("Preprocessing corpus")

# load spacy for preprocessing 
nlp_spacy = spacy.load("fr_core_news_lg")
spacy.require_gpu() # oops


mesures = {
    "centimères carrés": "cm²",
    "millimètres carrés": "mm²",
    "décimètres carrés": "dm²",
    "mètres carrés": "m²",
    "m2": "m²",
    "mètres cubes": "m³",
    "kilomètres": "km",
    "kilomètres carrés": "km²",
    "centimètres cubes": "km³",
    "décimètres cubes": "km³",
    "kilomètres cubes": "km³",
    "kilomètres par heure": "km/h",
}
pattern_mesures = r"\b(" + "|".join(re.escape(mesure) for mesure in mesures.keys()) + r")\b"


def normalize_token(token):

    if token.lemma_ == "pourcent" or token.lemma_ == "pourcents":
        return "%"
    elif token.lemma_ == "dollar" or token.lemma_ == "dollars":
        return "$"
    elif token.lemma_ == "euro" or token.lemma_ == "euros":
        return "€"
    elif str(token) == "n’" or token == "n'": # keep negates
        return "ne"
    else:
        puncts = re.compile("""[!"&\'()*+,\./:;<=>?@[\\]^_`{|}~]""")
        lemma = puncts.sub(' ', token.lemma_) # lemma without punct in it and replace by space
        
        return lemma


def apply_preprocessing(article):
    
    pos_to_exclude = ["PRON", "AUX", "DET", "PUNCT","OTHER", "SPACE"] 
    unwanted_tokens = ["de", "d'","d’","qu’","qu'","que", "au","à", "►", '!', '"', '#', '%', '&', \
                       "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', \
                       '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    tokens = []
    # some normalization before tokenizing
    article = re.sub("(?<!\w)(M\.)|(?<!\w)(m\.)", "monsieur", article) # do this before tokenizer splits M. into m
    article = re.sub("(Mme)|(?<!\w)(m\.)(mme)", "madame", article)
    article = re.sub(pattern_mesures, lambda match: mesures[match.group()], article)
    
    parsed_doc = nlp_spacy(article)
    
    for token in parsed_doc:
        token_str = str(token).strip("""-,+\/%()^*[]"'~;:!?.""")
    
        # ignore uwanted
        if token.pos_ in pos_to_exclude or str(token.lemma_) in unwanted_tokens:
            continue
            
        # if number
        if token_str.isdigit():
            # remove dots from nums
            tokens.append(token_str.replace(".",""))
            continue
                
        # keep only lexical words and lemmatized
        else:
            lemmas = normalize_token(token).split() # returns a list because we replace some punct by space

            if len(lemmas) == 0:
                continue
            lemmas = [token.lower().strip("""-,+\/%()^*[]"'~;:!?.""") for token in lemmas if token != "" and len(token) > 1] # no empty and smaller than 1
            
            tokens.extend(lemmas)
    return tokens


print("Tokenizing and lemmatizing texts...")

# is quite slow (~ 30- 60 mins)
preprocessed_corpus = (
    data.with_columns(
        pl.col('text_cleaned')
        .map_elements(apply_preprocessing)
        .alias("preprocessed_text")
    )
)

print("Done.")



def correct_names(tokens):
    tokens = tokens.to_list()
    for i, token in enumerate(tokens):
        if str(token) == "N":
            if str(tokens[i + 1]) == "VA":
                tokens[i + 1] = "N-VA"
                del tokens[i]
        elif str(token).lower() == "open":
            if str(tokens[i + 1]).lower() == "vld":
                tokens[i + 1] = "Open VLD"
                del tokens[i]
    return tokens


preprocessed_corpus_names = (
    preprocessed_corpus.with_columns(
        pl.when(pl.col('text_cleaned').str.contains("Open Vld|N-VA"))
        .then(pl.col("preprocessed_text").map_elements(correct_names))
        .otherwise("preprocessed_text")
    )
)


print_head("Writing to subcorpus to file rtbf_subcorpus.ndjson")
# write to file
preprocessed_corpus_names.write_ndjson("rtbf_subcorpus.ndjson")

print("Done.")



print_head("Preprocessing evaluation set")

eval_set_tokenized = (
    evaluation_sample.with_columns(
        pl.col('text_cleaned')
        .map_elements(apply_preprocessing)
        .alias("preprocessed_text")
    )
)
eval_set_tokenized = (
    eval_set_tokenized.with_columns(
        pl.when(pl.col('text_cleaned').str.contains("Open Vld|N-VA"))
        .then(pl.col("preprocessed_text").map_elements(correct_names))
        .otherwise("preprocessed_text")
    )
)

eval_set_tokenized.write_ndjson("evaluation_50_samples_tokenized.ndjson")
print("Done.")

