#!/usr/bin/env python
# coding: utf-8
#===============================================================================
#
#           FILE: preprocess_doc.py 
#         AUTHOR: Bianca Ciobanica
#          EMAIL: bianca.ciobanica@student.uclouvain.be
#
#           BUGS: 
#        VERSION: 3.11.4
#        CREATED: 22-03-2024 
#
#===============================================================================
#    DESCRIPTION: # function used to preprocess text 
#    
#   DEPENDENCIES:  
#
#          USAGE: python preprocess_doc.py
#===============================================================================

import sys
import os
import spacy
import re


# load spacy model
nlp_spacy = spacy.load("fr_core_news_md")

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


