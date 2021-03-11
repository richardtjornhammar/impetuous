"""
Copyright 2021 RICHARD TJÖRNHAMMAR

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
contact__ = "richard.tjornhammar@gmail.com"

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
    pdf.index = [v.replace(' ','') for v in  pdf.index.values]
    pdf_.index= [v.replace(' ','') for v in pdf_.index.values]
    dag_df = pd.concat([pdf.T,pdf_.T]).T
    return ( dag_df , tree )

def HierarchalEnrichment (
            analyte_df , dag_df , dag_level_label = 'DAG,l' ,
            ancestors_id_label = 'aid' , id_name = None , threshold = 0.05 ,
            p_label = 'C(Status),p', analyte_name_label = 'analytes' ,
            item_delimiter = ',' , alexa_elim=False , alternative = 'two-sided'
        ) :
    # BACKWARDS COMPATIBILITY
    return ( HierarchicalEnrichment (
            analyte_df=analyte_df , dag_df=dag_df , dag_level_label = dag_level_label ,
            ancestors_id_label = ancestors_id_label , id_name = id_name , threshold = threshold ,
            p_label = p_label , analyte_name_label = analyte_name_label ,
            item_delimiter = item_delimiter , alexa_elim = alexa_elim , alternative = alternative ) )

def HierarchicalEnrichment (
            analyte_df , dag_df , dag_level_label = 'DAG,l' ,
            ancestors_id_label = 'aid' , id_name = None , threshold = 0.05 ,
            p_label = 'C(Status),p', analyte_name_label = 'analytes' ,
            item_delimiter = ',' , alexa_elim=False , alternative = 'two-sided'
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
                pv,odds = group_significance( group , AllAnalytes=AllAnalytes, SigAnalytes=SigAnalytes , tolerance = threshold , alternative=alternative )
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
    df['Hierarchical,p'] = [ node_sig[idx] if idx in node_sig else 1. for idx in df.index.values ]
    df['Included analytes,ids'] = [ used_analytes[idx] if idx in used_analytes else '' for idx in df.index.values ]
    df = df.dropna()
    return ( df )

def calculate_hierarchy_matrix ( data_frame = None ,
                                 distance_matrix = None ,
                                 bVerbose = False ) :
    info__ = """ This is the saiga/pelican/panda you are looking for """
    from impetuous.clustering import connectivity , absolute_coordinates_to_distance_matrix
    import operator
    if not operator.xor( data_frame is None , distance_matrix is None ) :
        print ( "ONLY SUPPLY A SINGE DATA FRAME OR A DISTANCE MATRIX" )
        print ( "calculate_hierarchy_matrix FAILED" )
        exit(1)
    if not data_frame is None :
        if not 'pandas' in str(type(data_frame)) :
            print ( "ONLY SUPPLY A SINGE DATA FRAME WITH ABSOLUTE COORDINATES" )
            print ( "DATA MATRICES NEEDS TO BE SPECIFIED WITH \" distance_matrix = ... \" " )
            print ( "calculate_hierarchy_matrix FAILED" )
            exit(1)
        if bVerbose :
            print ( data_frame )
        distance_matrix = absolute_coordinates_to_distance_matrix(data_frame)

    if not distance_matrix is None :
        if bVerbose :
            print ( distance_matrix )

    uco_v = sorted(list(set(distance_matrix.reshape(-1))))
    hsers = []
    level_distance_lookup = {}
    for icut in range(len(uco_v)) :
        cutoff = uco_v[icut]
        # clustercontacts : clusterid , particleid relation
        # clustercontent : clusterid to number of particles in range
        clustercontent , clustercontacts = connectivity ( distance_matrix , cutoff ,
                                                        bVerbose=bVerbose )
        #
        # internal ordering is a range so this does not need to be a dict
        pid2clusterid = clustercontacts[:,0]
        level_distance_lookup['level'+str(icut)] = [ icut , cutoff , np.mean(clustercontent) ]
        hser = pd.Series(pid2clusterid,name='level'+str(icut),index=range(len(distance_matrix)))
        hsers.append(hser)
        if len(set(hser.values))==1:
            break
    if not data_frame is None:
        if 'pandas' in str(type(data_frame)):
            names = data_frame.index.values
    else:
        names = [ str(i) for i in range(len(distance_matrix)) ]
    res_df = pd.DataFrame ( hsers )
    res_df .columns = names

    hierarchy_matrix = res_df
    if bVerbose:
        print ()
        print ("TAKE NOTE THAT THE CLUSTER INDEXING BETWEEN LEVELS MIGHT NOT BE THE SAME")
        print ("YOU HAVE TO USE THE CLUSTER ID NUMBERS ACROSS A LEVEL TO DEDUCE WHICH PARTICLES")
        print ("BELONG TOGETHER IN A CLUSTER. THAT IS: I.E. CLUSTER 0 ON LEVEL 12 MIGHT NOT CORRESPOND")
        print ("TO CLUSTER 0 ON LEVEL 13. THE CALCULATED HIERARCHY MATRIX IS: ")
        print ( hierarchy_matrix )
        print ("AND THESE ARE THE DISTANCES THE HIERARCHY LEVELS CORRESPOND TO:" )
        print ( level_distance_lookup )
    return ( hierarchy_matrix , level_distance_lookup )


def parent_child_matrix_relationships ( hierarchy_matrix ,
                                        bVerbose = False ,
                                        bRemoveRedundant = True ,
                                        separators = ['_','-'] ) :

    print('WARNING: DEVELOPMENTAL VERSION')
    
    s_ = separators
    M = hierarchy_matrix
    ns,ms = np.shape(M)
    if not 'pandas' in str(type(M)):
        print ( "hierarchy_matrix MUST BE A PANDAS DATA FRAME" )
    if not ( len(set(M.index.values)) == ns and  len(set(M.columns.values)) == ms ):
        print( "USE UNIQUE COLUMN AND INDEX NAMES")
        exit(1)
    if bVerbose :
        print ( "THE hierarchy_matrix MUST BE ORDERED FROM THE LEAVES TO THE ROOT")
        print ( "LOWER INDICES THUS CORRESPONDS TO CHILDREN OF HIGHER ONES")
    n,c,levels = len(M),M.columns.values,M.index.values
    ancestor_offspring_relationships = []
    pc_df = None
    lookup = {}
    for i in range(n)[::-1][:-1]:
        I = i
        J = i-1
        parents  = M.T.groupby(M.iloc[I,:].values).apply(lambda x:x.index)
        children = M.T.groupby(M.iloc[J,:].values).apply(lambda x:x.index)
        parents_level_name = levels[I]
        children_level_name = levels[J]
        if bVerbose:
            print ( i )
            print ( parents.values , parents.index )
            print ( children.values , children.index )
        for p__,pidx in zip( parents.values , parents.index ):
            for c__,cidx in zip( children.values , children.index ):
                pcrel = []
                p_ = set(p__)
                c_ = set(c__)
                if len ( p_ & c_ ) > 0 and len( c_-p_) == 0 :
                    if bRemoveRedundant:
                        ps = '.'.join(p__)
                        cs = '.'.join(c__)
                        pcs= ps+'+'+cs
                        if not pcs in lookup:
                            lookup[pcs] = [(parents_level_name,pidx,I,children_level_name,cidx,J)]
                        else :
                            lookup[pcs] .append((parents_level_name,pidx,I,children_level_name,cidx,J))
                            continue
                    pcser = pd.Series( [ parents_level_name , pidx ,
                            children_level_name , cidx ] ,
                            index = ['Parent level label','Parent level cluster index',
                                     'Child level label','Child level cluster index'] ,
                            name = str(I)+s_[0]+str(pidx)+s_[1]+str(J)+s_[0]+str(cidx)  )
                    pcrel .append ( pd.DataFrame(pcser) )
                    if len ( pcrel ) > 0 :
                        if pc_df is None :
                            pc_df = pd.DataFrame(pcser)
                        else:
                            pc_df = pd.concat([pc_df.T,pd.DataFrame(pcser).T]).T
                        ancestor_offspring_relationships.append( pcrel )
    pc_df = pc_df.T
    if bRemoveRedundant:
        idx_rename = {}
        for item in lookup.items():
            if len(item[1])>1 :
                orig = str(item[1][0][2])  + s_[0] + str(item[1][0][ 1]) + s_[1] + \
                        str(item[1][0][-1])  + s_[0] + str(item[1][0][-2])
                new = str(item[1][0][2])  + s_[0] + str(item[1][0][ 1]) + s_[1] + \
                        str(item[1][-1][-1]) + s_[0] + str(item[1][-1][-2])
                if bVerbose :
                    print ( item )
                    print ( str(item[1][0][2])  + s_[0] + str(item[1][0][ 1]) )
                    print ( str(item[1][0][-1])  + s_[0] + str(item[1][0][-2]) )
                    print ( str(item[1][-1][-1]) + s_[0] + str(item[1][-1][-2]) )
                    print ( orig , new )
                    print ( pc_df.loc[orig,:])
                pc_df.loc[orig,'Child level label'] = item[1][-1][-3]
                pc_df.loc[orig,'Child level cluster index'] = item[1][-1][-2]
                idx_rename[orig] = new
        pc_df = pc_df.rename(index=idx_rename)
    return ( pc_df )

def create_cpgmt_lookup( pcdf , separators = ['_','-'] ):
        s_ = separators
        all_parents = list(set([v.split(s_[1])[0] for v in pcdf.index.values]))
        lookup      = {'content':['children','descriptions','parts']}
        children , descriptions , parts = [] , [] , []
        for parent in all_parents:
            selected    = pcdf.loc[ [idx for idx in pcdf.index.values if parent == idx.split(s_[1])[0]],:]
            for i_ in selected.index :
                a_level   = pcdf.loc[ i_ , [ c for c in pcdf.columns if 'label' in c ] ] .values[1]
                a_cluster = pcdf.loc[ i_ , [ c for c in pcdf.columns if 'index' in c ] ] .values[1]
                collected_parts = M.columns .values[ M.loc[a_level].values == a_cluster ]
                p_ = pcdf.loc[ i_ , [ c for c in pcdf.columns if 'label' in c ] ] .values[0] + s_[0]  + \
                     str(pcdf.loc[ i_ , [ c for c in pcdf.columns if 'index' in c ] ] .values[0])
                children    .append ( 'level'+i_.split(s_[1])[-1] )
                descriptions.append ( p_ )
                parts       .append ( collected_parts )
        lookup['children']     = children
        lookup['parts']        = parts
        lookup['descriptions'] = descriptions
        return ( lookup )

def write_cpgmt ( lookup ,
                  filename = 'childparentparts.gmt',
                  bVerbose = False ) :
    if bVerbose :
        print ( """
                   If you want to check specific level clusters
                   using traditional methods such as GSEA or
                   perhaps try my hierarchical enrichment or
                   awesome righteuous-fa method ...
                   irregardless you'll need to create a gmt file """ )
    if 'content' in lookup :
        with open( filename , 'w' ) as of :
            print ( '\n'.join( [ '\t'.join([c,d,'\t'.join(p)]) for \
                                 (c,d,p) in zip( *[ lookup[ c ] for c in lookup['content'] ]) ]) ,
                    file=of )

if __name__ == '__main__' :

    if True :
        #
        bVerbose = False
        if bVerbose:
            print ( "For creating pc and gmt files" )
            print ( "note that the description includes the parent as a reference" )
            print ( "to create a pc file you should reorder i.e. using" )
            print ( "$ cat childparentparts.gmt | awk '{print $2,$1}'" )
        #
        pdf   = pd.read_csv( '../data/genes.tsv' , '\t' , index_col=0 )
        M , L = calculate_hierarchy_matrix ( pdf )
        cpgl  = create_cpgmt_lookup( parent_child_matrix_relationships ( M ) , separators = ['_','-'] )
        write_cpgmt ( cpgl )

    if True :
        print ( "hierarchy matrix test"  )
        R = np.random.rand(90).reshape(30,3)
        P = np.zeros(90).reshape(30,3)
        P [ 1:10 ,: ] += 1
        P [ 0:5  ,: ] += 2
        P [ 20:  ,: ] += 3
        P [ 15:20,: ] += 2
        P [ 25:  ,: ] += 2

        pdf = pd.DataFrame( P + R ,
                       index    = ['pid'+str(i) for i in range(30)] ,
                       columns  = ['x','y','z'] )

        M,L = calculate_hierarchy_matrix ( pdf )
        print ( M )
        parent_child_matrix_relationships ( M )

        from impetuous.visualisation import *
        X,Y,W,Z = [],[],[],[]
        for item in L.items():
            X.append(item[1][1])
            W.append(item[1][1])
            Z.append(item[1][2])
            Y.append(len(set(M.loc[item[0]].values)))
        from bokeh.plotting import show
        #
        # SHOW THE COORDINATION AND SEGREGATION FUNCTIONS
        # BOTH ARE WELL KNOWN FROM STATISTICAL PHYSICS
        #
        show ( plotter( [X,W] , [Y,Z] ,
               [nice_colors[0],nice_colors[2]] ,
               legends = ['segregation','coordination'],
               axis_labels = ['distance','Number']) )
