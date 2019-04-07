"""
Copyright 2019 RICHARD TJÖRNHAMMAR

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import pandas as pd
import numpy as np
import json
import sys

def read_conversions(file_name):
    gene2ens = {} ; non_unique = []
    with open( file_name , 'r' ) as infile:
        if sys.version_info[0] < 3:
            infile.next()
        else :
            next(infile)
        for line in infile:
            words = line.strip().split('\t')
            if len(words)==2 :
                ens, gene = words
                if gene in gene2ens:
                    gene2ens[gene].append(ens)
                else :
                    gene2ens[gene] = [ens]
    return gene2ens

def read_gene_ensemble_conversion(file_name):
    gene2ens = {} ; non_unique = []
    with open(file_name,'r') as infile:
        if sys.version_info[0] < 3:
            infile.next()
        else:
            next(infile)
        for line in infile:
            words = line.strip().split('\t')
            if len(words)==2 :
                ens, gene = words
                if gene in gene2ens:
                    non_unique.append((gene,ens,gene2ens[gene]))
                else :
                    gene2ens[gene] = ens
        if len(non_unique)>0:
            print(' WARNING ' )
            print( 'FOUND ', len(non_unique), ' NON UNIQUE ENTRIES' )
    return gene2ens

def create_synonyms( convert_file , unique_mapping=False ):
    # CREATE SYNONYMS
    ens2sym , sym2ens = {} , {}
    if unique_mapping:
        sym2ens = read_gene_ensemble_conversion( convert_file )
        ens2sym = { v:k for k,v in sym2ens.items() }
    else :
        sym2ens_list = read_conversions( convert_file )
        ens2sym_list = {}
        for s,L in sym2ens_list.items() :
            for e in L:
                if e in ens2sym_list:
                    ens2sym_list.append(s)
                else:
                    ens2sym_list[e] = [s]
        ens2sym = ens2sym_list
        sym2ens = sym2ens_list
    return ( ens2sym , sym2ens )

def flatten_dict( s2e ) :
    #
    # FORCES FIRST ELEMENT
    #
    ndict = {}
    for (s,e) in s2e.items() :
        if 'list' in str(type(e)) :
            ndict[s] = e[0]
        else :
            ndict[s] = e
    return ( ndict )


if __name__ == '__main__' :

    base = '../../../data/'
    convert_file = base + 'naming_and_annotations/conv.txt'
    ens2sym , sym2ens = create_synonyms( convert_file )
    s2e = { **flatten_dict(sym2ens) }

    name,col,sep = 'Networks.nfo',0,'\t'
    with open(name,'r') as input:
        for line in input:
            gname = line.split(sep)[col].replace('\n','')
            if gname in s2e:
                print(gname)
            else:
                print('MISSING:',gname)
