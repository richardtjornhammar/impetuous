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
from impetuous.quantification import group_significance
from impetuous.convert import *

def pathway_frame_from_file ( filename , 
        delimiter = '\t' , item_sep = ',' ) :
    pdf = None
    with open( filename,'r' ) as input :
        for line in input :
            lspl = line.replace('\n','').split(delimiter)
            analytes_ = lspl[2:]
            desc = lspl[1]
            iden = lspl[0]
            ps = pd.Series( [desc , item_sep.join(analytes_) ] , 
                        name = iden , index = ['description','analytes'] )
            pdf = pd.concat([pdf,pd.DataFrame(ps).T])
    return ( pdf )

def create_dag_representation_df ( pathway_file = '../data/GROUPDEFINITIONS.gmt' , 
                                   pcfile = '../data/PCLIST.txt' 
                                 ) :
    pc_list_file = pcfile
    tree , ance , desc = parent_child_to_dag ( pc_list_file )
    pdf = make_pathway_ancestor_data_frame ( ance )
    pdf_ = pathway_frame_from_file( pathway_file )
    dag_df = pd.concat([pdf.T,pdf_.T]).T
    return ( dag_df , tree )

def HierarchalEnrichment (
            analyte_df , dag_df , dag_level_label = 'DAG,l' ,
            ancestors_id_label = 'aid' , id_name = None , threshold=0.05 ,
            p_label = 'C(Status),p', analyte_name_label = 'analytes',
            item_delimiter = ',' , alexa_elim=False
        ) :
    #
    # NEEDS AN ANALYTE SIGNIFICANCE FRAME:
    #     INCLUDING P VALUES OF ANALYTES
    # DAG GRAPH DESCRIPTION FRAME:
    #     INCLUDING NODE ID, NODE ANALYTES FIELD (SEPERATED BY ITEM DELIMITER)
    #     INCLUDING ANCESTORS FIELD (SEPERATED BY ITEM DELIMITER)
    #     DAG LEVEL OF EACH NODE
    tolerance = threshold
    df = dag_df ; dag_depth = np.max( df[dag_level_label].values )
    AllAnalytes = set( analyte_df.index.values ) ; nidx = len( AllAnalytes )
    SigAnalytes = set( analyte_df.iloc[ (analyte_df.loc[:,p_label].values < tolerance), : ].index.values )
    if len( AllAnalytes ) == len( SigAnalytes ) :
        print ( 'THIS STATISTICAL TEST WILL BE NONSENSE' )
        print ( 'TRY A DIFFERENT THRESHOLD' )
    marked_analytes = {} ; used_analytes = {} ; node_sig = {}
    for d in range( dag_depth, 0, -1 ) : 
        # ROOT IS NOT INCLUDED
        filter_ = df [ dag_level_label ] == d
        nodes = df.iloc[ [i for i in np.where(filter_)[ 0 ]] ,:].index.values
        for node in nodes :
            if 'nan' in str(df.loc[node,analyte_name_label]).lower() :
                continue
            analytes_ = df.loc[node,analyte_name_label].replace('\n','').replace(' ','').split(item_delimiter)
            try :
                group = analyte_df.loc[[a for a in analytes_ if a in AllAnalytes] ].dropna( axis=0, how='any', thresh=analyte_df.shape[1]/2 ).drop_duplicates()
            except KeyError as e :
                continue
            if node in marked_analytes :
                unused_group = group.loc[ list( set(group.index.values)-marked_analytes[node] ) ]
                group = unused_group
            L_ = len( group ) ; str_analytes=','.join(group.index.values)
            if L_ > 0 :
                used_analytes[node] = ','.join( group.index.values )
                pv,odds = group_significance( group , AllAnalytes=AllAnalytes, SigAnalytes=SigAnalytes , tolerance = threshold )
                node_sig[node] = pv ; marked_ = set( group.index.values )
                ancestors = df.loc[node,ancestors_id_label].replace('\n','').replace(' ','').split(item_delimiter)
                if alexa_elim and pv > threshold : # USE ALEXAS ELIM ALGORITHM : IS NOT DEFAULT
                    continue
                for u in ancestors :
                    if u in marked_analytes :
                        us = marked_analytes[u]
                        marked_analytes[u] = us | marked_
                    else :
                        marked_analytes[u] = marked_
    df['Hierarchal,p'] = [ node_sig[idx] if idx in node_sig else 1. for idx in df.index.values ]
    df['Included analytes,ids'] = [ used_analytes[idx] if idx in used_analytes else '' for idx in df.index.values ]
    df = df.dropna()
    return ( df )
