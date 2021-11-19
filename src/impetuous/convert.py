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

import pandas as pd
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import json
import sys

import typing

class Node ( object ) :
    def __init__ ( self ) :
        self.id_          :str   = ""
        self.label_       :str   = ""
        self.description_ :str   = ""
        self.level_       :int   = 0      # NODES ARE MYOPIC
        self.metrics_     :list  = list()
        self.links_       :list  = list()
        self.ascendants_  :list  = list() # INWARD LINKS  , DIRECT ASCENDENCY  ( 1 LEVEL )
        self.descendants_ :list  = list() # OUTWARD LINKS , DIRECT DESCENDENCY ( 1 LEVEL )
        self.data_        :dict  = dict() # OTHER THINGS SHOULD BE ALL INFORMATION FLOATING IN USERSPACE

    def can_it_be_root(self) -> bool :
        return ( len(self.ascendants) == 0 )

    def supplement ( self, n:super ) -> None :
        self.label_       = n.label_
        self.description_ = n.description_
        self.level_       = n.level_
        self.metrics_     = [ *self.metrics_     , *n.metrics_     ]
        self.links_       = [ *self.links_       , *n.links_       ]
        self.ascendants_  = [ *self.ascendants_  , *n.ascendants_  ]
        self.descendants_ = [ *self.descendants_ , *n.descendants_ ]
        self.data_        = { **self.data_, **n.data_ }

    def assign_all ( self, identification : str ,
                           links : type(list(str())) ,
                           label : str = "" ,
                           description : str = "" ) -> object :
        self.set_id( identification )
        self.add_label( label )
        self.add_description( description )
        self.add_links( links , bClear=True )
        return ( self )

    def get_data ( self ) -> dict :
        return ( self.data_ )

    def overwrite_data ( self, data:dict ) -> None :
        self.data_ = data

    def set_id ( self, identification:str ) -> None :
        self.id_ = identification

    def add_label ( self, label : str ) -> None :
        self.label_ = label

    def add_description ( self, description : str ) -> None :
        self.description_ = description

    def identification ( self ) -> str :
        return ( self.id_ )

    def label ( self ) -> str :
        return ( self.label_ )

    def description ( self ) -> str :
        return ( self.description_ )

    def add_link ( self, identification:str , bClear:bool = False , linktype:str = 'links' ) -> None :
        edges = self.get_links( linktype )
        if bClear :
            edges = list()
        edges .append( identification )
        self.links_ = list(set( edges ))
    #
    # NOTE : list[str] type declaration is not working in Python3.8
    def add_links ( self, links:type(list(str())), bClear:bool = False , linktype:str = 'links' ) -> None :
        edges = self.get_links( linktype )
        if bClear :
            edges = list()
        for e in links :
            edges .append ( e )
        self.links_ = list(set( edges ))

    def get_links ( self , linktype:str='links' ) -> type(list(str())) :
        if not linktype in set([ 'links' , 'ascendants' , 'descendants' ]):
            print ( ' \n\n!!FATAL!!\t' + ', '.join([ 'links' , 'ascendants' , 'descendants' ]) \
                  + '\t ARE THE ONLY VALID EDGE TYPES (linktype)' )
            exit ( 1 )
        if linktype == 'links' :
            return ( self.links_ )
        if linktype == 'ascendants' :
            return ( self.ascendants_  )
        if linktype == 'descendants' :
            return ( self.descendants_ )

    def show ( self ) -> None :
        s_inf = "NODE [" + self.identification() \
                   + "," + self.label() + "] - " \
                   + self.description() + "\nEDGES:"
        for linktype in [ 'links' , 'ascendants' , 'descendants' ] :
            s_inf += '\n['+linktype+'] : '
            for l in self.get_links(linktype=linktype) :
                s_inf += l + '\t'
        print ( s_inf )

class NodeGraph ( Node ) :
    # https://github.com/richardtjornhammar/RichTools/commit/c4b9daa78f2a311995d142b0e74fba7c3fdbed20#diff-0b990604c2ec9ebd6f320ebe92099d46e0ab8e854c6e787fac2f208409d112d3
    def __init__( self ) :
        self.root_id_       = ''
        self.desc_          = "SUPPORTS DAGS :: NO STRUCTURE ASSERTION"
        self.num_edges_     = 0
        self.num_vertices_  = 0
        self.graph_map_     = dict()

    def keys ( self )   -> list :
        return( self.graph_map_.keys() )

    def values ( self ) -> list :
        return( self.graph_map_.values() )

    def items ( self )  -> list :
        return( self.graph_map_.values() )

    def get_node ( self, nid : str ) -> Node :
        return ( self.graph_map_[nid] )

    def set_root_id ( self, identification : str ) -> None :
        self.root_id_ = identification

    def get_root_id ( self ) -> str :
        return ( self.root_id_ )

    def add ( self, n : Node ) -> None :
        if n.identification() in self.graph_map_ :
            self.graph_map_[ n.identification() ].supplement( n )
        else :
            self.graph_map_[ n.identification() ] = n
            if len ( self.graph_map_ ) == 1 :
                self.set_root_id( n.identification() )

    def get_dag ( self ) -> dict() :
        return ( self.graph_map_ )

    def show ( self ) -> None :
        print ( self.desc_ )
        for item in self.get_dag().items() :
            print ( '\n' + item[0] + '::' )
            item[1].show()

    def complete_lineage ( self , identification : str ,
                           order:str    = 'depth'      ,
                           linktype:str = 'ascendants' ) -> dict :
        # 'ascendants' , 'descendants'
        root_id = identification
        results = self.search( order=order , root_id=identification , linktype=linktype )
        results['path'] = [ idx for idx in results['path'] if not idx==identification ]
        return ( results )

    def search ( self , order:str = 'breadth', root_id:str = None , linktype:str = 'links' ) -> dict :
        path:list   = list()
        visited:set = set()
        if root_id is None :
            root_id = self.get_root_id()
        S:list      = [ root_id ]
        if not order in set(['breadth','depth']) :
            print ( 'order MUST BE EITHER breadth XOR depth' )
            exit ( 1 )

        if order == 'breadth' :
            while ( len(S)>0 ) :
                v = S[0] ; S = S[1:]
                ncurrent:Node = self.get_node(v)
                visited       = visited|set([v])
                path.append( ncurrent.identification() )
                links         = ncurrent.get_links(linktype)
                for w in links :
                    if not w in visited and len(w)>0:
                        S.append( w ) # QUE

        if order == 'depth' :
            while ( len(S)>0 ) :
                v = S[0] ; S = S[1:]
                if not v in visited and len(v)>0 :
                    visited       = visited|set([v])
                    ncurrent:Node = self.get_node(v)
                    links         = ncurrent.get_links(linktype)
                    for w in links :
                        if not w in visited and len(w)>0:
                            S = [*[w],*S] # STACK
                    path.append( ncurrent.identification() )

        return ( { 'path':path , 'order':order , 'linktype':linktype } )


def add_attributes_to_node_graph ( p_df:type(pd.DataFrame) , tree:NodeGraph ) -> NodeGraph :
    for idx in p_df.index.values :
        for attribute in p_df.columns.values :
            tree.get_node( idx ).get_data()[attribute] = p_df.loc[idx,attribute]
    return ( tree )


def ascendant_descendant_to_dag ( relationship_file:str = './PCLIST.txt' ,
                                  i_a:int = 0 , i_d:int = 1 ,
                                  identifier:str = None , sep:str = '\t' ) -> NodeGraph :
    RichTree = NodeGraph()
    with open(relationship_file,'r') as input :
        for line in input :
            if not identifier is None :
                if not identifier in line :
                    continue
            lsp = line.split(sep)
            ascendant  = lsp[i_a].replace('\n','')
            descendant = lsp[i_d].replace('\n','')

            n = Node()
            n.set_id(ascendant)
            n.add_label("")
            n.add_description("")
            n.add_links([descendant],linktype='links'       )
            n.add_links([descendant],linktype='descendants' )

            m = Node()
            m.set_id(descendant)
            m.add_label("")
            m.add_description("")
            m.add_links([ascendant],linktype='links'      )
            m.add_links([ascendant],linktype='ascendants' )

            RichTree.add(n)
            RichTree.add(m)

    all_names   = RichTree.keys()
    descendants = [ ( idx , set( RichTree.complete_lineage( idx,linktype='descendants')['path'] ) ) for idx in all_names ]
    ancestors   = [ ( idx , set( RichTree.complete_lineage( idx,linktype='ascendants' )['path'] ) ) for idx in all_names ]

    return ( RichTree , ancestors , descendants )


def drop_duplicate_indices( df ):
    df_ = df.loc[~df.index.duplicated(keep='first')]
    return df_

def write_tree( tree , outfile='tree.json' ):
    print ( 'WARNING::LEGACY' )
    root = [ eid for eid,ancestor in tree.in_degree() if ancestor == 0 ][ 0 ]
    o_json = json_graph.tree_data( tree , root )
    if not outfile is None:
        with open(outfile, 'w') as o_file:
            json.dump(o_json, o_file )
    return( o_json )

def add_attributes_to_tree ( p_df , tree ):
    add_attributes_to_node_graph ( p_df , tree )
    return ( tree )

def parent_child_to_dag (
             relationship_file:str = './PCLIST.txt' ,
             i_p:int = 0 , i_c:int = 1 , identifier:str = None
           ) :

    return ( ascendant_descendant_to_dag ( relationship_file = relationship_file,
                                      i_a = i_p , i_d = i_c , identifier=identifier ) )

def make_pathway_ancestor_data_frame(ancestors):
    p_df = None
    for k,v in ancestors :
        t_df = pd.DataFrame([[','.join(list(v)),len(v)]],index=[k],columns=['DAG,ancestors','DAG,level'])
        if p_df is None :
            p_df = t_df
        else :
            p_df = pd.concat([p_df,t_df])
    return( p_df )

def normalise_for_apples_and_oranges_stats( X , method='ordinal' ):
    X_ = rankdata( X , method=method )/len(X)
    return(X_)

def make_group_analytes_unique( grouping_file , delimiter='\t' ):
    uniqe_grouping_file = '/'.join(grouping_file.split('/')[:-1]) + '/unique_'+grouping_file.split('/')[-1]
    with open( uniqe_grouping_file , 'w' ) as of:
        with open( grouping_file ) as input :
            for line in input :
                vline = line.replace('\n','').split(delimiter)
                gid, gdesc, analytes_ = vline[0], vline[1], list(set(vline[2:]))
                nvec = [gid,gdesc] ; [ nvec.append(a) for a in analytes_ ]
                print ( delimiter.join(nvec) , file = of )

def read_conversions(file_name) :
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
    # FORCES FIRST ELEMENT
    ndict = {}
    for (s,e) in s2e.items() :
        if 'list' in str(type(e)) :
            ndict[s] = e[0]
        else :
            ndict[s] = e
    return ( ndict )

def convert_rdata_to_dataframe ( filename ) :
    #
    from rpy2.robjects import r as R
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    import rpy2.robjects as ro
    #
    print ( 'WARNING THIS PROGRAM NEED VALUE ERROR CHECKING' )
    rd_ = R.load( filename )
    if 'matrix' in str( type( R[rd_[0]] ) ).lower() :
        column_names = [ R[rd_[0]].colnames ]
        index_names  = [ R[rd_[0]].rownames ]
    else :
        column_names = [ [r for r in _rd_.colnames] for _rd_ in R[rd_[0]]]
        index_names  = [ [r for r in _rd_.rownames] for _rd_ in R[rd_[0]]]
    #
    pandas2ri.activate()
    #
    # SMALL HELPER FUNCTION THAT TRANSFORMS A RDATA OBJECT INTO
    # A PANDAS DATAFRAME. CURRENTLY THERE IS NO VALUE ERROR CHECKING
    #
    rd = R.load( filename )
    raw_df_l = []
    if 'ndarray' in str( type( R[rd[0]] ) ).lower() :
        [ raw_df_l.append( R[rd[0]] ) ]
    else :
        [ raw_df_l.append( rdf ) for rdf in ro.vectors.DataFrame(R[rd[0]]) ]
    full_df_dict = {} ; i_ = 0
    for raw_df,colnames,rownames in zip( raw_df_l,column_names,index_names ) :
        pdf = pd.DataFrame( raw_df , columns=colnames , index=rownames )
        full_df_dict[i_] = pdf
        i_ = i_ + 1
    pandas2ri.deactivate()
    return ( full_df_dict )

def read_xyz(fname='argon.xyz') :
    df_ = pd.read_csv('./argon.xyz',header=2,sep=' ')
    vals = df_.columns.values
    df_.loc[len(df_)] = vals
    df_.columns = ['A','X','Y','Z']
    return ( df_.apply(pd.to_numeric) )


import os
if __name__ == '__main__' :
    #
    bMOFA_data = True
    if bMOFA_data :
        import os
        # os.system('mkdir ../data')
        # os.system('wget https://github.com/bioFAM/MOFAdata/blob/master/data/CLL_data.RData')
        # os.system('mv CLL_data.RData ../data/.')

        df_dict = convert_rdata_to_dataframe( filename = '../data/CLL_data.RData' )
        pruned_df = df_dict[2].T.dropna().T
        journal_df = df_dict[3].loc[['IGHV'],:]
        mask = [ v>=0 for v in journal_df.loc['IGHV'].values ]
        journal_df = journal_df.iloc[ :,mask ]
        use = list(set(pruned_df.columns)&set(journal_df.columns))
        analyte_df =  pruned_df .loc[ :,use ].apply( lambda x:np.log2(1+x) )
        journal_df = journal_df .loc[ :,use ]

        print ( analyte_df,journal_df )
        from impetuous.quantification import *
        qdf = quantify_groups ( analyte_df , journal_df , 'anova~C(IGHV)' , '~/Work/data/naming_and_annotations/reactome/reactome.gmt' )
        print(qdf)
        exit(1)

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

    n = Node()
    n.set_id("richard")
    n.add_label("eating")
    n.add_description("rice")
    n.add_links(["cola","soysauce"])
    n.show()

    RichTree = NodeGraph()
    nodeid = "0"; label = "2"; v_ids = ["1","6"]
    RichTree.add( Node().assign_all( nodeid,v_ids,label ) )
    nodeid = "1"; label = "7"; v_ids = ["2","3"]
    RichTree.add( Node().assign_all( nodeid,v_ids,label ) )
    nodeid = "2"; label="2"; v_ids=["",""];
    RichTree.add( Node().assign_all( nodeid,v_ids,label ) )
    nodeid = "3"; label="6"; v_ids=["4","5"];
    RichTree.add( Node().assign_all( nodeid,v_ids,label ) )
    nodeid = "4"; label="5"; v_ids=["",""];
    RichTree.add( Node().assign_all( nodeid,v_ids,label ) )
    nodeid = "5"; label="11"; v_ids=["",""];
    RichTree.add( Node().assign_all( nodeid,v_ids,label ) )
    nodeid = "6"; label="5"; v_ids=["7",""];
    RichTree.add( Node().assign_all( nodeid,v_ids,label ) )
    nodeid = "7"; label="9"; v_ids=["8",""];
    RichTree.add( Node().assign_all( nodeid,v_ids,label ) )
    nodeid = "8"; label="4"; v_ids=["",""];
    RichTree.add( Node().assign_all( nodeid,v_ids,label ) )
    nodeid = "9"; label="3"; v_ids=["",""];
    RichTree.add( Node().assign_all( nodeid,v_ids,label ) )
    nodeid = "10"; label="1"; v_ids=["",""];
    RichTree.add( Node().assign_all( nodeid,v_ids,label ) )
    
    #RichTree.show()
    print ( "ROOT::", RichTree.get_root_id() )
    route = RichTree.search( root_id='0', order='breadth' )
    print ( "ROUTE:: " , route )
    route = RichTree.search( root_id='0', order='depth' )
    print ( "ROUTE:: " , route )    

                
             
