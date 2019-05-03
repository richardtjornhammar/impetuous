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

def HierarchalEnrichment ( 
            analyte_df , dag_df , dag_level_label = 'DAG,l' ,
            ancestors_id_label = 'aid' , id_name = None , threshold=0.05 ,
            p_label = 'C(Status),p', analyte_name_label = 'analytes',
            item_delimiter = ','
        ) :
    #
    # NEEDS AN ANALYTE SIGNIFICANCE FRAME:
    #     INCLUDING P VALUES OF ANALYTES
    # DAG GRAPH DESCRIPTION FRAME:
    #     INCLUDING NODE ID, NODE ANALYTES FIELD (SEPERATED BY ITEM DELIMITER)
    #     INCLUDING ANCESTORS FIELD (SEPERATED BY ITEM DELIMITER)
    #     DAG LEVEL OF EACH NODE
    df = dag_df ; dag_depth = df[dag_level_label].apply(np.max)
    AllAnalytes = set( analyte_df.index.values ) ; nidx = len( AllAnalytes )
    SigAnalytes = set( analyte_df.iloc[ (analyte_df.loc[:,p_label].values < tolerance), : ].index.values )
    if len( AllAnalytes ) == len( SigAnalytes ) :
        print ( 'THIS STATISTICAL TEST WILL BE NONSENSE' )
    marked_analytes = {} ; used_analytes = {}
    for d in range( dag_depth, 0, -1 ): # ROOT IS NOT INCLUDED
        filter_ = df [ dag_level_label ] == d
        nodes = np.where( filter_ )[ 0 ]
        for node in nodes :
            analytes_ = df[ node,analyte_name_label ].values.replace('\n','').replace(' ','').split(item_delimiter)
            try :
                group = analyte_df.loc[[a for a in analytes_ if a in AllAnalytes] ].dropna( axis=0, how='any', thresh=analyte_df.shape[1]/2 ).drop_duplicates()
            except KeyError as e :
                continue
            if node in marked_analytes :
                unused_group = group.loc[ list( set(group.index.values)-marked_analytes[node] ) ]
            else :
                unused_group = group
            L_ = len( unused_group ) ; str_analytes=','.join(unused_group.index.values)
            if L_ > 0 :
                used_analytes[node] = ','.join( group.index.values )
                pv,odds = group_significance( group , AllAnalytes=AllAnalytes, SigAnalytes=SigAnalytes )
                node_sig[node] = pv ; marked_ = set(group.index.values)
                ancestors = df[node,ancestors_id_label].values.replace('\n','').replace(' ','').split(item_delimiter)
                for u in ancestors :
                    if u in marked_analytes :
                        us = marked_analytes[u]
                        marked_analytes[u] = us | marked_
                    else :
                        marked_analytes[u] = marked_
    df['Hierarchal,p'] = [ node_sig[idx] for idx in df.index.values ]
    df['Included analytes,ids'] = [ used_analytes[idx] for idx in df.index.values ]

    return ( df )

