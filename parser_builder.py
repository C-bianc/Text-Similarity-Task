#!/usr/bin/env python
# ~* coding: utf-8 *~
#===============================================================================
#
#           FILE: create_parser.py 
#         AUTHOR: Bianca Ciobanica
#          EMAIL: bianca.ciobanica@student.uclouvain.be
#
#           BUGS: 
#        VERSION: 3.11.4
#        CREATED: 20-05-2024 
#
#===============================================================================
#    DESCRIPTION: package used for creating parser and checking arguments (date) 
#    
#   DEPENDENCIES:  
#
#          USAGE: python parser_builder.py 
#===============================================================================
import polars as pl
import argparse
from datetime import datetime

def create_parser():
    parser = argparse.ArgumentParser(description="Programe cherchant les documents similaires à partir d'un texte donné en entrée.",
                                     prog="")

    # input argument options
    input_group = parser.add_mutually_exclusive_group(required=True) # input text and index not allowed at same time
    input_group.add_argument('-text', '-texte', '-t', type=str, help='Le programme nécessite d\'un texte en entrée englobé de guillements. \n Exemple : "<texte long>"')
    input_group.add_argument('-i', '-id', type=int, help='Index du document à chercher. La liste se trouve dans all_ids.txt')

    # publication date argument options
    date_group = parser.add_argument_group('Date : options')
    date_group.add_argument('-d', type=int, help='Jour de publication (chiffres)')
    date_group.add_argument('-m', type=int, help='Mois de publication (chiffres)')
    date_group.add_argument('-y', type=int, help='Année de publication (chiffres). Doit être entre 2008-2021.')

    # is between options
    # source : https://www.wrighcters.io/reading-date-arguments-to-a-python-script-using-argparse/
    date_range_group = parser.add_argument_group('Date : options entre deux dates (début --start-date, fin --end-date)')
    date_range_group.add_argument('--start-date', type=lambda s: datetime.strptime(s, '%d-%m-%Y'), help='Date de début (format: DD-MM-YYY)')
    date_range_group.add_argument('--end-date', type=lambda s: datetime.strptime(s, '%d-%m-%Y'), help='Date de fin (format: DD-MM-YYYY)')
    
    # top_n similar docs argument
    parser.add_argument('-n', type=int, default=3, help='Nombre <n> de documents similaires à afficher.')

    return parser

def check_date(args, parser):

    if (args.y or args.m or args.d) and (args.start_date or args.end_date):
        parser.error("Impossible d'utiliser une option individuelle pour la date en même temps qu'une option sur une durée (start & end)")

    day, month, year = args.d, args.m, args.y
    
    if month:
        if 1 > month or month > 12:
            parser.error("Veuillez entrer un nombre entre 1 et 12 pour le mois. ;)")

    if day:
        if 1 > day or day > 31:
            parser.error("Veuillez entrer un nombre entre 1 et 31 pour le jour. ;)")

    if year:
        if year != 2021 and year != 2020:
            parser.error("Le sous-corpus comprend seulement les articles publiés entre 2020 et 2021 (inclus)")

    # check date
    if args.start_date and args.end_date:
        if args.start_date >= args.end_date:
            parser.error('La date de début doit être antérieure à la date de fin.')

        elif args.start_date.year not in [2020,2021] or args.end_date.year not in [2020,2021]:
            parser.error("Le sous-corpus comprend seulement les articles publiés entre 2020 et 2021 (inclus)")

        elif 1 > args.start_date.month or args.start_date.month > 12:
            parser.error("Veuillez entrer un nombre entre 1 et 12 pour le mois. ;)")
        
        elif 1 > args.start_date.day or args.start_date.day > 31:
            parser.error("Veuillez entrer un nombre entre 1 et 31 pour le jour. ;)")

