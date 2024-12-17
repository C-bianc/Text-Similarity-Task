#!/usr/bin/env python
# coding: utf-8
#===============================================================================
#
#           FILE: launch_session.py 
#         AUTHOR: Bianca Ciobanica
#          EMAIL: bianca.ciobanica@student.uclouvain.be
#
#           BUGS: 
#        VERSION: 3.11.4
#        CREATED: 20-05-2024 
#
#===============================================================================
#    DESCRIPTION:  Programme interactif qui permet de faire une recherche 
#                  de similarité sur le contenu textuel de documents ou d'un texte
#                  donné en entrée. L'utilisateur peut également choisir une date
#                  Le programme stocke également les résultats sous format csv
#    
#   DEPENDENCIES:  pip install -r requirements.txt
#
#          USAGE: python launch_session.py
#===============================================================================

print("Chargement...\n")

from get_most_sim import process_query
from parser_builder import *
from gensim.models import KeyedVectors
import pickle
import shlex # for parsing quoted strings
import random

def load_data():
    # load corpus vectors
    with open('final_w2v_corpus_vectors.pkl', 'rb') as f:
        w2v_corpus_vectors = pickle.load(f)

    # load w2v model
    w2v_model = KeyedVectors.load("model/w2v_vectors_v200.kv", mmap=None)

    # load corpus
    corpus = pl.read_ndjson("rtbf_subcorpus.ndjson")

    return  w2v_model, w2v_corpus_vectors, corpus
    

### Interactive part ###
def get_user_input():
    print("1. Trouver des documents similaires à partir d'un texte")
    print("2. Quitter ce super chouette programme.")
    print("3. Une citation de Bob Ross ¯\_(ツ)_/¯")
    print("Insérez 'quitter' ou 'exit' pour arrêter le programme à tout instant.\n")

    choice = input("Choisissez votre option : ")
    return choice

bob_ross_quotes = [
    "We don't make mistakes, just happy little accidents.",
    "There are no limits here, you start out by believing here.",
    "Talent is a pursued interest. Anything that you're willing to practice, you can do.",
    "Let's get crazy.",
    "You need the dark in order to show the light.",
    "We want happy paintings. Happy paintings. If you want sad things, watch the news.",
    "In painting, you have unlimited power. You have the ability to move mountains. You can bend rivers."
]


if __name__ == "__main__":
    print("__________________________________________")
    print("Bienvenue ! Vous avez lancé le programme qui permet de trouver des documents similaires sur leur contenu textuel.\nProgramme créé par Bianca.")
    print("Le programme enregistrera les résultats dans le dossier 'resultats_requetes' pour chaque requête.\n")
    print("Entrer le choix de votre action.\n")

    while True:
        choice = get_user_input().strip()
        parser = create_parser()

        if choice == '1':
            print("\n\n---------- Usage -------------\n")
            parser.print_help()

            while True:
                user_input = input("\nChoisissez vos arguments. (ex: -text \"du texte\" -d 12 -m 5 -y 2020 -n 5)\n").strip()

                if user_input.lower() in ['quitter', 'exit', 'quit']:
                    print("À bientôt !")
                    exit()

                if user_input.lower() in ['help', '-h', 'aide', '--help']:
                    parser.print_help()
                    continue

                else:
                    # parse user arguments
                    try :
                        user_args = shlex.split(user_input)

                    # if no double quotations
                    except ValueError as e:
                        if "No closing quotation" in str(e):
                            print("\nVeuillez mettre le texte en double guillements : \"texte\"")
                            continue

                    # do the thing
                    try:
                        args = parser.parse_args(user_args)
                        check_date(args, parser)
                    except SystemExit as e:
                        print()
                        continue

                    ### Similarity results ###
                    try:
                        model, vectors, corpus = load_data()
                        process_query(args, parser, model, vectors, corpus)

                    except Exception as e:
                        print(e)
                        continue

        elif choice == '2' or choice.lower() in ['quitter', 'exit']:
            print("À bientôt !")
            break

        elif choice == '3':
            print("__________________________________________")
            print("\nBob Ross says: ", random.choice(bob_ross_quotes), "\n")
            print("__________________________________________")
            next

        else:
            print("__________________________________________")
            print("\nVeuillez choisir entre 1 ou 2 ou 3.\n")

