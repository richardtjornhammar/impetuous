"""
Copyright 2023 RICHARD TJÖRNHAMMAR

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


def create_dag_representation_df ( pathway_file:str = '../data/GROUPDEFINITIONS.gmt' ,
                                   pcfile:str = '../data/PCLIST.txt', identifier:str='R-HSA',
                                   item_sep:str = ','
                                 ) :
    pc_list_file = pcfile
    tree , ance , desc = parent_child_to_dag ( pc_list_file , identifier = identifier )
    pdf_ = pathway_frame_from_file( pathway_file )
    root = tree.get_root_id()
    lost = set(tree.keys()) - set(pdf_.index.values.tolist())
    for l in lost :
        pdf_.loc[l] = ['NA','']
    DF = []
    for idx in pdf_.index.values :
        ord_ascendants  = tree.search(  root_id         = idx                   ,
                                        linktype        = 'ascendants'          ,
                                        order           = 'depth'               )['path']
        level = len ( ord_ascendants ) - 1
        ord_descendants = tree.search(  root_id         = idx                   ,
                                        linktype        = 'descendants'         ,
                                        order           = 'depth'               )['path']
        ord_ascendants  = [ a for a in ord_ascendants  if not ( a==idx ) ]
        ord_descendants = [ d for d in ord_descendants if not ( d==idx ) ]
        if len( ord_ascendants ) > 0 :
            parent = ord_ascendants[0]
        else:
            parent = ''
        dag_anscestors          = item_sep.join(  ord_ascendants )
        dag_descendants         = item_sep.join( ord_descendants )
        DF.append( [level,dag_anscestors,dag_descendants,parent , *pdf_.loc[idx].values ] )
    ndag_df = pd.DataFrame( DF ,
                  columns = [*['DAG,level','DAG,ancestors','DAG,descendants','DAG,parent'],*pdf_.columns.values],
                  index   = pdf_.index.values )
    return ( ndag_df , tree )


def fill_in_full_hierarchy ( df:pd.DataFrame    = None ,
                         gmtfile:str        = '/home/USER/data/Reactome/reactome_v71.gmt' ,
                         pcfile :str        = '/home/USER/data/Reactome/NewestReactomeNodeRelations.txt',
                         fields:list[float] = ['p-value'] ) -> pd.DataFrame :
    dag_df , tree      = create_dag_representation_df ( pathway_file = gmtfile , pcfile = pcfile )
    for field in fields :
        dag_df.loc[:,field] = 1.
    for idx in df.index.values :
        dag_df.loc[ idx , fields ] = df.loc[ idx , fields ]
    dag_df.loc[:,'parent'] = dag_df.loc[:,'DAG,parent']
    for idx in dag_df.index.values :
        if idx in set(df.index.values) :
            dag_df.loc[idx,fields] = df.loc[idx,fields]
    dag_df.loc[:,'name'] = dag_df.index.values
    return ( dag_df )


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

def hierarchicalEnrichment (
            analyte_df:pd.DataFrame , dag_df:pd.DataFrame , dag_level_label:str = 'DAG,l' ,
            ancestors_id_label:str = 'aid' , id_name:str = None , threshold:float = 0.05 ,
            p_label:str = 'C(Status),p', analyte_name_label:str = 'analytes' ,
            item_delimiter:str = ',' , alexa_elim:bool=False , alternative:str = 'two-sided',
            test_type:str = 'fisher', bNoMasking:bool=False, bOnlyMarkSignificant:bool=False
        )  -> pd.DataFrame :
    return ( HierarchicalEnrichment (
            analyte_df=analyte_df , dag_df=dag_df , dag_level_label = dag_level_label ,
            ancestors_id_label = ancestors_id_label , id_name = id_name , threshold = threshold ,
            p_label = p_label, analyte_name_label = analyte_name_label ,
            item_delimiter = item_delimiter , alexa_elim = alexa_elim , alternative = alternative ,
            test_type = test_type, bNoMasking = bNoMasking, bOnlyMarkSignificant=bOnlyMarkSignificant ) )

def HierarchicalEnrichment (
            analyte_df:pd.DataFrame , dag_df:pd.DataFrame , dag_level_label:str = 'DAG,l' ,
            ancestors_id_label:str = 'aid' , id_name:str = None , threshold:float = 0.05 ,
            p_label:str = 'C(Status),p', analyte_name_label:str = 'analytes' ,
            item_delimiter:str = ',' , alexa_elim:bool=False , alternative:str = 'two-sided',
            test_type:str = 'fisher', bNoMasking:bool=False, bOnlyMarkSignificant:bool=False
        )  -> pd.DataFrame  :
    #
    # NEEDS AN ANALYTE SIGNIFICANCE FRAME:
    #     INCLUDING P VALUES OF ANALYTES
    # DAG GRAPH DESCRIPTION FRAME:
    #     INCLUDING NODE ID, NODE ANALYTES FIELD (SEPERATED BY ITEM DELIMITER)
    #     INCLUDING ANCESTORS FIELD (SEPERATED BY ITEM DELIMITER)
    #     DAG LEVEL OF EACH NODE
    from impetuous.special import unpack
    all_annotated = set( [ w for w in unpack( [ str(v).split(item_delimiter)\
                for v in dag_df.loc[:,analyte_name_label ].values.reshape(-1)\
                        if not 'nan' == str(v).lower() ]) ])
    tolerance = threshold
    df = dag_df ; dag_depth = np.max( df[dag_level_label].values )
    AllAnalytes = set( analyte_df.index.values ) ; nidx = len( AllAnalytes )
    SigAnalytes = set( analyte_df.iloc[ (analyte_df.loc[:,p_label].values < tolerance), : ].index.values )
    if len( AllAnalytes ) == len( SigAnalytes ) :
        print ( 'THIS STATISTICAL TEST WILL BE NONSENSE' )
        print ( 'TRY A DIFFERENT THRESHOLD' )
    marked_analytes = {} ; used_analytes = {} ; node_sig = {}; node_odds = {}
    for d in range( dag_depth, 0, -1 ) :
        # ROOT IS NOT INCLUDED
        filter_ = df [ dag_level_label ] == d
        nodes = df.iloc[ [i for i in np.where(filter_)[ 0 ]] ,:].index.values
        for node in nodes :
            if 'nan' in str(df.loc[node,analyte_name_label]).lower() :
                continue
            analytes_ = df.loc[node,analyte_name_label].replace('\n','').replace(' ','').split(item_delimiter)
            try :
                group = analyte_df.loc[[a for a in analytes_ if a in AllAnalytes] ]
            except KeyError as e :
                continue
            if node in marked_analytes :
                unused_group = group.loc[ list( set(group.index.values)-marked_analytes[node] ) ]
                group = unused_group
            L_ = len( group ) ; str_analytes=','.join(group.index.values)
            if L_ > 0 :
                used_analytes[node] = ','.join( group.index.values )
                pv , odds = group_significance( group ,
				AllAnalytes = AllAnalytes, SigAnalytes = SigAnalytes , AllAnnotated=all_annotated ,
				tolerance = threshold , alternative=alternative, TestType=test_type )
                node_sig[node] = pv ; node_odds[node] = odds ; marked_ = set( group.index.values )
                if bOnlyMarkSignificant : # NOT DEFAULT
                    marked_ = marked_ & SigAnalytes
                ancestors = df.loc[node,ancestors_id_label].replace('\n','').replace(' ','').split(item_delimiter)
                if ( alexa_elim and pv > threshold ) or bNoMasking :  # USE ALEXAS ELIM ALGORITHM : IS NOT DEFAULT
                    continue
                for u in ancestors :
                    if u in marked_analytes :
                        us = marked_analytes[u]
                        marked_analytes[u] = us | marked_
                    else :
                        marked_analytes[u] = marked_
    df['Hierarchical,p']	= [ node_sig[idx] if idx in node_sig else 1. for idx in df.index.values ]
    df['Hierarchical,odds']	= [ node_odds[idx] if idx in node_sig else 1. for idx in df.index.values ]
    df['Included analytes,ids'] = [ used_analytes[idx] if idx in used_analytes else '' for idx in df.index.values ]
    df = df.dropna()
    return ( df )

def groupFactorHierarchicalEnrichment (
            analyte_df:pd.DataFrame , journal_df:pd.DataFrame , formula:str ,
            dag_df:pd.DataFrame = None , gmtfile:str = None , pcfile:str = None ,
            dag_level_label:str = 'DAG,level' , ancestors_id_label:str = 'DAG,ancestors' ,
            id_name:str = None , threshold:float = 0.05 , bVerbose:bool = False ,
            p_label:str = 'C(Status),p', analyte_name_label:str = 'analytes' ,
            item_delimiter:str = ',' , alexa_elim:bool=True , alternative:str = 'two-sided' ,
            test_type:str = 'fisher', bNoMasking:bool=False , agg_function=lambda x:x[0]) -> pd.DataFrame :
    # https://github.com/richardtjornhammar/righteous/commit/6c63dcc922eb389237220bf65ffd4b1fa3241a2c
    #
    # A HIERARCHICALLY CORRECTED GROUP FACTOR ANALYSIS METHOD !
    #
    from impetuous.quantification import find_category_interactions , find_category_variables, qvalues
    from sklearn.decomposition import PCA
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    #
    if dag_df is None :
        from impetuous.hierarchical import create_dag_representation_df
        dag_df,tree = create_dag_representation_df( pathway_file = gmtfile , pcfile = pcfile , item_sep = item_delimiter )
    if bVerbose:
        print ( dag_df )
    dimred = PCA()
    statistical_formula = formula
    eval_df = None
    cats = []
    for c in find_category_variables(formula):
        cs_ = list( set( journal_df.loc[c].values ) )
        journal_df.loc[c+',str'] = journal_df.loc[c]
        journal_df.loc[c] = [ { c_:i_ for i_,c_ in zip( range(len(cs_)),cs_ ) }[v] for v in journal_df.loc[c].values ]
        cats.append(c)
    vars = [ v.replace(' ','') for v in formula.split('~')[1].split('+') if np.sum([ c in v for c in cats ])==0 ]
    #
    # NEEDS AN ANALYTE FRAME
    #       A  JOURNAL FRAME
    # DAG GRAPH DESCRIPTION FRAME :
    #     INCLUDING NODE ID, NODE ANALYTES FIELD (SEPERATED BY ITEM DELIMITER)
    #     INCLUDING ANCESTORS FIELD (SEPERATED BY ITEM DELIMITER)
    #     DAG LEVEL OF EACH NODE
    #
    from impetuous.special import unpack
    all_annotated = set( [ w for w in unpack( [ str(v).split(item_delimiter)\
                for v in dag_df.loc[:,analyte_name_label ].values.reshape(-1)\
                        if not 'nan' == str(v).lower() ]) ])
    tolerance = threshold
    df = dag_df ; dag_depth = np.max( df[dag_level_label].values )
    AllAnalytes = set( analyte_df.index.values ) & all_annotated ; nidx = len( AllAnalytes )
    marked_analytes = {} ; used_analytes = {} ; node_sig = {}; node_odds = {}
    for d in range( dag_depth, 0, -1 ) :
        # ROOT IS NOT INCLUDED
        filter_ = df [ dag_level_label ] == d
        nodes = df.iloc[ [i for i in np.where(filter_)[ 0 ]] ,:].index.values
        for node in nodes :
            if 'nan' in str(df.loc[node,analyte_name_label]).lower() :
                continue
            analytes_ = df.loc[node,analyte_name_label] .replace('\n','').replace(' ','').split(item_delimiter)
            try :
                group = analyte_df.loc[ [a for a in analytes_ if a in AllAnalytes] ]
            except KeyError as e :
                continue
            if node in marked_analytes :
                unused_group = group.loc[ list( set(group.index.values)-marked_analytes[node] ) ]
                group = unused_group
            L_ = len( group ) ; str_analytes=','.join(group.index.values)
            if L_ > 0 :
                gid = node
                used_analytes[node] = ','.join( group.index.values )
                if bVerbose :
                    print ( gid )
                Xnew = dimred.fit_transform(group.T.values)
                group_expression_df = pd.DataFrame([Xnew.T[0]],columns=analyte_df.columns.values,index=['Group'])
                cdf = pd.concat( [group_expression_df,journal_df] ).T
                cdf = cdf.loc[ : , ['Group',*vars,*cats]  ].apply(pd.to_numeric)
                linear_model = ols( 'Group~' + formula.split('~')[1], data = cdf ).fit()
                table = sm.stats.anova_lm(linear_model,typ=2 )
                rdf = group_expression_df
                for idx in table.index.values :
                    for jdx in table.loc[idx].index :
                        rdf[ idx + ';' + jdx.replace('PR(>F)','Hierarchical,p')] = table.loc[idx].loc[jdx]
                rdf ['Hierarchical,p']          = agg_function( table.iloc[:,-1].values )
                rdf ['description']		= df.loc[node,'description']+','+str(L_)
                rdf ['Included analytes,ids']	= str_analytes
                rdf .index = [ gid ]
                pv = rdf['Hierarchical,p'].values
                if eval_df is None :
                    eval_df = rdf
                else :
                    eval_df = pd.concat([eval_df,rdf])
                marked_ = set( group.index.values )
                ancestors = df.loc[node,ancestors_id_label].replace('\n','').replace(' ','').split(item_delimiter)
                if ( alexa_elim and pv > threshold ) or bNoMasking :  # USE ALEXAS ELIM ALGORITHM : IS DEFAULT
                    continue
                for u in ancestors :
                    if u in marked_analytes :
                        us = marked_analytes[u]
                        marked_analytes[u] = us | marked_
                    else :
                        marked_analytes[u] = marked_
    edf = eval_df.T.fillna(1.0)
    for col in eval_df.columns :
        if ',p' in col :
            q = [q_[0] for q_ in qvalues(edf.loc[col,:].values)]; l=col.split(',')[0]+',q'
            edf.loc[l] = q
    return ( edf.T )

def hierarchy_matrix ( distance_matrix:np.array   = None ,
                       coordinates:np.array       = None ,
                       linkage_distances:np.array = None ) -> dict :
    from impetuous.clustering import connectivity , absolute_coordinates_to_distance_matrix
    import operator
    if not operator.xor( coordinates is None , distance_matrix is None ) :
        print ( "ONLY COORDINATES OR A DISTANCE MATRIX" )
        print ( "calculate_hierarchy_matrix FAILED" )
        print ( "DATA MATRICES NEEDS TO BE SPECIFIED WITH \" distance_matrix = ... \" " )
        exit(1)
    if not coordinates is None :
        distance_matrix = absolute_coordinates_to_distance_matrix(coordinates)

    nmt_ = np.shape(distance_matrix)
    if linkage_distances is None :
        uco_v = sorted(list(set(distance_matrix.reshape(-1))))
    else :
        uco_v = sorted(list(set(linkage_distances.reshape(-1))))

    level_distance_lookup = {}
    hsers = []
    for icut in range(len(uco_v)) :
        cutoff = uco_v[icut]
        # clustercontacts : clusterid , particleid relation
        # clustercontent : clusterid to number of particles in range
        #from clustering import connectivity # LOCAL TESTING
        clustercontent , clustercontacts = connectivity ( distance_matrix , cutoff )
        #
        # internal ordering is a range so this does not need to be a dict
        level_distance_lookup[icut] = [ icut , cutoff , np.mean(clustercontent) ]
        hsers.append(clustercontacts[:,0])
        if len( set(clustercontacts[:,0]) ) == 1 : # DON'T DO HIGHER COMPLETE SYSTEM VALUES
            break
    return ( { 'hierarchy matrix':np.array(hsers) , 'lookup':level_distance_lookup} )

def reformat_hierarchy_matrix_results ( hierarchy_matrix:np.array , lookup:dict=None ) :
    CL = {}
    for i in range(len(hierarchy_matrix)):
        row = hierarchy_matrix[i]
        d = i
        if not lookup is None :
            d   = lookup[i][1]
        sv_ = sorted(list(set(row)))
        cl  = {s:[] for s in sv_}
        for i in range( len( row ) ) :
            cl[row[i]].append(i)
        for v_ in list( cl.values() ) :
            if tuple(v_) not in CL :
                CL[ tuple(v_) ] = d
    return ( CL )

def build_snn_hierarchy ( distm:np.array ):
    #
    # print ( "BUILDING SYMMETRY BROKEN HIERARCHY" )
    if True :
        nnvec                   = []
        hierarchy_matrix        = []
        N                       = np.min( np.shape(distm) )
        from impetuous.clustering import nearest_neighbor_graph_matrix , connectivity_boolean
        for NN in range ( 1 , N ) :
            nnvec .append( NN )
            NN_bool     = nearest_neighbor_graph_matrix ( distm , NN )[0]<np.inf
            solution    = connectivity_boolean ( NN_bool )
            hierarchy_matrix .append( [ cid[0] for cid in solution[1] ] )
            if len(solution[0])>1 :
                continue
            else :
                break
    else :
        print ( "DONE AT" , N )
    return ( hierarchy_matrix )

def calculate_hierarchy_matrix ( data_frame = None ,
                                 distance_matrix = None ,
                                 bVerbose = False,
                                 coarse_grain_structure = 0 ) :

    print ( "WARNING:LEGACY METHOD. CHECK IF hierarchy_matrix MIGHT SUIT YOU BETTER" )

    info__ = """ This is the saiga/pelican/panda you are looking for RICEARD"""
    # print (info__ )
    from impetuous.clustering import connectivity , absolute_coordinates_to_distance_matrix
    import operator
    if not operator.xor( data_frame is None , distance_matrix is None ) :
        print ( "ONLY SUPPLY A SINGE DATA FRAME OR A DISTANCE MATRIX" )
        print ( "calculate_hierarchy_matrix FAILED" )
        print ( "DATA MATRICES NEEDS TO BE SPECIFIED WITH \" distance_matrix = ... \" " )
        exit(1)
    if not data_frame is None :
        if not 'pandas' in str(type(data_frame)) :
            print ( "ONLY SUPPLY A SINGE DATA FRAME WITH ABSOLUTE COORDINATES" )
            print ( "DATA MATRICES NEEDS TO BE SPECIFIED WITH \" distance_matrix = ... \" " )
            print ( "calculate_hierarchy_matrix FAILED" )
            exit ( 1 )
        if bVerbose :
            print ( data_frame )
        distance_matrix = absolute_coordinates_to_distance_matrix(data_frame.values)

    nmt_ = np.shape(distance_matrix)
    if nmt_[0] != nmt_[1] : # SANITY CHECK
        print ( "PLEASE SUPPLY A PROPER SQUARE DISTANCE MATRIX" )
        print ( "DATA MATRICES NEEDS TO BE SPECIFIED WITH \" distance_matrix = ... \" " )
        print ( "from scipy.spatial.distance import squareform , pdist\nabsolute_coordinates_to_distance_matrix = lambda Q:squareform(pdist(Q))" )

    if not distance_matrix is None :
        if bVerbose :
            print ( distance_matrix )

    if bVerbose :
        print ( "EMPLOYING SORTING HAT" )
    uco_v = sorted(list(set(distance_matrix.reshape(-1))))
    if coarse_grain_structure>0 :
        if bVerbose :
            nuco  = len(uco_v)
            print ( "WILL COARSE GRAIN THE HIERARCHY STRUCTE INTO" )
            print ( "AT MAX:", np.ceil(nuco/coarse_grain_structure), " LEVELS" )
            print ( "TECH: NTOT >", nuco,",\t dN >", coarse_grain_structure )
            uco_v = uco_v[::coarse_grain_structure]

    level_distance_lookup = {}

    if bVerbose :
        print ( "DOING CONNECTIVITY CLUSTERING" )

    hsers = []
    for icut in range(len(uco_v)) :
        cutoff = uco_v[icut]
        # clustercontacts : clusterid , particleid relation
        # clustercontent : clusterid to number of particles in range
        #from clustering import connectivity # LOCAL TESTING
        clustercontent , clustercontacts = connectivity ( distance_matrix , cutoff ,
                                                          bVerbose = bVerbose )
        #
        # internal ordering is a range so this does not need to be a dict
        pid2clusterid = clustercontacts[:,0]
        level_distance_lookup['level'+str(icut)] = [ icut , cutoff , np.mean(clustercontent) ]
        hser = pd.Series(pid2clusterid,name='level'+str(icut),index=range(len(distance_matrix)))
        hsers.append(hser)
        if bVerbose :
            print ( 100.0*icut/len(uco_v) ," % DONE ")

        if len(set(hser.values)) == 1 :
            break
    if not data_frame is None :
        if 'pandas' in str(type(data_frame)):
            names = data_frame.index.values
    else :
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
                                        separators = ['_','-'],
                                        iLegacy = 1 ) :
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
    if iLegacy == 1 :
        return ( pc_df )
    else :
        return ( pc_df , hierarchy_matrix )


def create_cpgmt_lookup ( pcdf , hierarchy_matrix , separators = ['_','-'] ):
        s_ = separators
        M  = hierarchy_matrix
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

#
# TOOLS LOCAL
def ordered_remove ( str,delete ):
    for d in delete :
        str = str.replace(d,'')
    return ( str )

def error ( criterion,message ) :
    if criterion :
        print ( message )
        exit ( 1 )

def build_pclist_word_hierarchy ( filename = None ,  # 'new_compartment_genes.gmt',
                                  ledger   = None ,
    delete   = ['\n'] , group_id_prefix = None,
    analyte_prefix  = 'ENSG', root_name = 'COMP0000000000',
    bReturnList = False , bUseGroupPrefix = False ,
    bSingleChild = False , bSingleDescent = True ) :

    bSingleChild = bSingleChild or bSingleDescent # USER SHOULD SET THE bSingleDescent OPTION

    bUseFile = not filename is None
    import operator
    error ( not operator.xor ( filename is None , ledger is None ), "YOU MUST SUPPLY A GMT FILE XOR A DICTIONARY" )
    if bUseFile :
        error ( not '.gmt' in filename , 'MUST HAVE A VALID GMT FILE' )
    #
    # RETURNS THE PC LIST THAT CREATES THE WORD HIERARCHY
    # LATANTLY PRESENT IN THE GMT ANALYTE (GROUPING) DEFINITIONS
    #
    S_M = set()
    D_i = dict()

    bUseGroupPrefix = not group_id_prefix is None
    if bUseGroupPrefix :
        bUseGroupPrefix = 'str' in str(type(group_id_prefix)).lower()
    check_prefix = analyte_prefix
    if bUseGroupPrefix :
        check_prefix = group_id_prefix

    if bUseFile :
        with open ( filename,'r' ) as input :
            for line in input :
                lsp = ordered_remove(line,delete).split('\t')
                if not check_prefix in line :
                    continue
                S_i = set(lsp[2:])
                D_i [ lsp[0] ] = tuple( (lsp[1] , S_i , len(S_i)) )
                S_M = S_M | S_i
    else :
        for item in ledger.items() :
            print(item)
            if bUseGroupPrefix :
                if not check_prefix in item[0]:
                    continue
            else :
                if not check_prefix in ''.join(item[1][1]):
                    continue
            S_i = set( item[1][1] )
            D_i [ item[0] ] = tuple( (item[1][0] , S_i , len(S_i)) )
            S_M = S_M | S_i

    isDecendant  = lambda sj,sk : len(sj-sk)==0
    relative_idx = lambda sj,sk : len(sk-sj)

    parent_id = root_name
    parent_words = S_M

    all_potential_parents = [ [root_name,S_M] , *[ [ d[0],d[1][1]] for d in D_i.items() ] ]

    PClist = []
    CPlist = {}
    for parent_id,parent_words in all_potential_parents:
        lookup    = {}
        for d in D_i .items() :
            if isDecendant ( d[1][1] , parent_words ) :
                Nij = relative_idx ( d[1][1] , parent_words  )
                if Nij in lookup :
                    lookup[Nij] .append(d[0])
                else :
                    lookup[Nij] = [d[0]]
        ledger = sorted ( lookup.items() )

        for ie_ in range( len( ledger ) ) :
            l1 = ledger[ ie_ ][0]
            for potential_child in ledger[ie_][1]:
                pchild_words  = D_i[ potential_child ][1]
                bIsChild      = True
                if potential_child == parent_id :
                    bIsChild  = False
                    break
                check         = [ je_ for je_ in range( ie_ + 1 )] [::-1]
                if len(check) > 0 :
                    for je_ in check :
                        l2 = ledger[ je_ ][0]
                        for relative in ledger[je_][1] :
                            if D_i[relative][0] == D_i[potential_child][0] :
                                continue
                            relative_words = D_i[relative][1]
                            bIsChild = len(relative_words^pchild_words)>0 or (len(relative_words^pchild_words)==0 and l2==l1 )
                            if not bIsChild :
                                break
                if bIsChild :
                    if potential_child in CPlist :
                        if CPlist[potential_child][-1]>relative_idx(pchild_words,parent_words):
                            CPlist[potential_child] = [parent_id , potential_child , relative_idx(pchild_words,parent_words) ]
                    else :
                        CPlist[potential_child] = [parent_id , potential_child , relative_idx(pchild_words,parent_words) ]
                    PClist .append ( [parent_id , potential_child ] )
    D_i[root_name] = tuple( ('full cell',S_M,len(S_M)) )
    pcl_ = []

    if bSingleChild:
        PClist = [ (v[0],v[1]) for k,v in CPlist.items() ]
    if bReturnList :
        return ( [PClist,D_i] )
    else :
        return ( PClist )



def prune_lvl_pclist(asc:list) -> list :
    cp = dict()
    for n,p,c in asc :
        if c in cp :
            l = cp[c][1]
            if n<l :
                cp[c]=[p,n]
        else :
            cp[c] = [p,n]

    pc_pruned=list() 
    for item in cp.items():
        pc_pruned.append([item[1][0],item[0]])
    return ( pc_pruned )

def dict_to_pclist( groups:dict , L0:int=None , bPruned:bool=True ) -> list :
    # SIMPLIFIED LOGIC FOR CREATING A PARENT CHILD LIST
    # FROM A COLLECTION OF SAIGA PROOF WORDS.
    PC = list()
    if L0 is None :
        L0 = 1E15
    gS = groups.keys()
    for gm in gS :
        L = L0*10
        for gn in gS :
            if gn==gm :
                continue
            A,B = set(groups[gm]) , set(groups[gn])
            if len(A-B) >= 0 and len(B-A) == 0 :
                l = len(A-B)
                PC.append([ l, gm , gn ] )
    if bPruned :
        return ( prune_lvl_pclist(PC) )
    return ( PC )

def simple_gmtfile_to_pclist( gmtname:str , gmt_delimiter:str = '\t' , bPruned:bool=True ) -> list :
    groups		=	dict()
    gmt_delimiter	=	'\t'
    L0	= 0
    with open ( gmtname , 'r' ) as input :
        for line in input :
            info = line.replace('\n','').split(gmt_delimiter)
            groups[info[0]] = info[2:]
            if len(set(info[2:]))>L0:
                L0 = len(set(info[2:]))
    return ( dict_to_pclist( groups , L0 , bPruned=bPruned )  )



def matrixZ2linkage_dict_tuples ( Z :np.array ) -> dict :
    # from scipy.cluster.hierarchy import linkage
    from scipy.cluster.hierarchy import fcluster
    CL = {}
    for d in Z[:,2] :
        row = fcluster ( Z ,d, 'distance' )
        sv_ = sorted(list(set(row)))
        cl  = {s:[] for s in sv_}
        for i in range( len( row ) ) :
            cl[row[i]].append(i)
        for v_ in list( cl.values() ) :
            CL[tuple(v_)] = d
    return ( CL )


if __name__ == '__main__' :

    if False :
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

    if False :
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


    if True :
        # https://github.com/richardtjornhammar/impetuous/blob/728bef88f1bba64a051603b807e06231269c7dbb/new_compartment_genes.gmt
        PClist,D_i = build_pclist_word_hierarchy (  filename = 'new_compartment_genes.gmt', delete   = ['\n'],
               group_id_prefix = 'COMP', analyte_prefix  = 'ENSG', root_name = 'COMP0000000000', bReturnList=True )

        for pc in PClist :
            print ( '\t'.join( pc ) )
            show_leftward_dependance = lambda s1,s2:[len(s1-s2),len(s1),len(s2)]
            print ( D_i[pc[0]][0], D_i[pc[1]][0] )
            print ( show_leftward_dependance( D_i[pc[0]][1],D_i[pc[1]][1]) )

