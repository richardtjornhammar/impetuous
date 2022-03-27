"""
Copyright 2022 RICHARD TJÖRNHAMMAR

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

    def is_a_root ( self, n:int=1 ) -> bool :
        return ( len( self.ascendants_  ) < n )

    def is_a_leaf( self, n:int=1 ) -> bool :
        return ( len( self.descendants_ ) < n )
    
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
        # ASSIGNS ALL META DATA AND BIPOLAR LINKS
        self.set_id( identification )
        self.add_label( label )
        self.add_description( description )
        self.add_links( links , bClear=True )
        return ( self )

    def set_level ( self,level:int ) -> None :
        self.level_ = level

    def set_metrics ( self , metrics:list ) -> None :
        self.metrics_ = [ *self.metrics_ , *metrics ]

    def get_metrics ( self ) -> list :
        return ( self.metrics_ )

    def level ( self ) -> None :
        return ( self.level_ )

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
        for item in self.get_data().items() :
            s_inf += '\n'+str(item[0])+'\t'+str(item[1])
        print ( s_inf )

class Neuron ( Node ) :
    def __init__ ( self ) :
        #
        # DEV
        # A BIOLOGICAL PHYSICS NEURON
        # AS NODE BUT ALSO
        #
        self.region_      :str   = ""
        self.strength_    :float = 0
        self.reactivity_  :float = 0
            
    def activation_(self,stimulus:float) -> None :
        return ( None )
    
    def pot_(self,stimulus:float) -> None :
        # POTENTIATE
        return ( None )
    
    def dep_(self,stimulus:float) -> None :
        # DEPRESS
        return ( None )

 
class NodeGraph ( Neuron ) :
    #
    # ONLY DEPENDENT ON FUNCTIONALITY COMPATIBLE WITH THE NODE CLASS
    # EVEN IF IT INHERITS HIGHER TYPES
    # EXTEND THE NODEGRAPH FOR EACH ADDITIONAL NODE TYPE
    # IN NEW GRAPH TYPES. ALWAYS DEPENDS ON THE MOST EXTENDED SUBTYPE 
    # SO THAT NODE < NEURON < OTHER HIGHER TYPES
    #
    # https://github.com/richardtjornhammar/RichTools/commit/c4b9daa78f2a311995d142b0e74fba7c3fdbed20#diff-0b990604c2ec9ebd6f320ebe92099d46e0ab8e854c6e787fac2f208409d112d3
    def __init__( self ) :
        self.root_id_          = ''
        self.desc_             = "SUPPORTS DAGS :: NO STRUCTURE ASSERTION"
        self.num_edges_        = 0
        self.num_vertices_     = 0
        self.graph_map_        = dict()
        self.adjacency_matrix_ = dict() 

    def keys ( self )   -> list :
        return ( self.graph_map_.keys() )

    def values ( self ) -> list :
        return ( self.graph_map_.values() )

    def items ( self )  -> list :
        return ( self.graph_map_.items() )

    def list_roots ( self ) ->  type(list(str())) :
        roots = [] # BLOODY ROOTS
        for name,node in self.items():
            if node.is_a_root() :
                roots.append( name )
        return ( roots )

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

    def get_dag ( self ) -> dict :
        return ( self.graph_map_ )

    def get_graph ( self ) -> dict :
        return ( self.graph_map_ )

    def show ( self ) -> None :
        print ( self.desc_ )
        for item in self.get_dag().items() :
            print ( '\n' + item[0] + '::' )
            item[1].show()

    def complete_lineage ( self , identification : str ,
                           order:str    = 'depth'      ,
                           linktype:str = 'ascendants' ) -> dict :
        root_id = identification
        results = self.search ( order=order , root_id=identification , linktype=linktype )
        results['path'] = [ idx for idx in results['path'] if not idx==identification ]
        return ( results )

    def retrieve_leaves ( self , identification : str  ,
                           order:str    = 'depth'      ,
                           linktype:str = 'descendants' ) -> dict :
        root_id = identification      
        results = self.search ( order=order , root_id=identification ,
                                linktype=linktype, bOnlyLeafNodes=True )
        results['path'] = [ idx for idx in results['path'] if not idx==identification ]        
        return ( results )

    def search ( self , order:str = 'breadth' , root_id:str = None ,
                 linktype:str = 'links' , stop_at:str = None ,
                 bOnlyLeafNodes:bool = False ) -> dict :
        #
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
                if not bOnlyLeafNodes or ncurrent.is_a_leaf() :
                    path.append( ncurrent.identification() )
                #
                # ADDED STOP CRITERION FOR WHEN THE STOP NODE IS FOUND
                if not stop_at is None :
                    if stop_at == v :
                        S = []
                        break
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
                    if not bOnlyLeafNodes or ncurrent.is_a_leaf() :
                        path.append( ncurrent.identification() )
                    #
                    # ADDED STOP CRITERION FOR WHEN THE STOP NODE IS FOUND
                    if not stop_at is None :
                        if stop_at == v :
                            S = []
                            break

        return ( { 'path':path , 'order':order , 'linktype':linktype } )
    
    def connectivity ( self, distm:np.array , alpha:float , n_connections:int=1 , bOld:bool=True ) -> list :
        #
        # AN ALTERNATIVE METHOD
        # DOES THE SAME THING AS THE CONNECTIVITY CODE IN MY
        # CLUSTERING MODULE (in src/impetuous/clustering.py )
        # OR IN https://github.com/richardtjornhammar/RichTools/blob/master/src/cluster.cc
        # ADDED TO RICHTOOLS HERE: https://github.com/richardtjornhammar/RichTools/commit/74b35df9c623bf03570707a24eafe828f461ed90#diff-25a6634263c1b1f6fc4697a04e2b9904ea4b042a89af59dc93ec1f5d44848a26
        # CONNECTIVITY SEARCH FOR (connectivity) CONNECTIVITY
        #
        # THIS ROUTINE RETURNS A LIST BELONGING TO THE CLUSTERS
        # WITH THE SET OF INDICES THAT MAPS TO THE CLUSTER
        # THIS METHOD IS NOW ALSO IN THE clustering.py MODULE
        # AND IS CALLED connectedness
        # THIS CLASS WILL EMPLOY THE JIT connectivity IMPLEMENTATION
        # IN THE FUTURE BECAUSE IT IS SUPERIOR
        #
        if len ( distm.shape ) < 2 :
            print ( 'PLEASE SUBMIT A SQUARE DISTANCE MATRIX' )
            exit(1)
        #
        if bOld : # WATER CLUSTERING ALGO FROM 2009
            from impetuous.clustering import connectivity as connections
            results = connections ( distm , alpha )
            L = [set() for i in range(len(results[0]))]
            for c in results[1]:
                L[c[0]] = L[c[0]]|set([c[1]])
            return ( L )
        #
        def b2i ( a:list ) -> list :
            return ( [ i for b,i in zip(a,range(len(a))) if b ] )
        def f2i ( a:list,alf:float ) -> list :
            return ( b2i( a<=alf ) )
        L = []
        for a in distm :
            bAdd = True
            ids = set( f2i(a,alpha) )
            for i in range(len(L)) :
                if len( L[i]&ids ) >=  n_connections :
                    L[i] = L[i] | ids
                    bAdd = False
                    break
            if bAdd and len(ids) >= n_connections :
                L .append( ids )
        return ( L )

    def linkages_to_pclist ( self , links:dict ) -> list :
        bottoms_up = sorted([ (v,k) for k,v in links.items()])
        PClist = []
        while ( len(bottoms_up)>1 ) :
            child = bottoms_up[0]
            for parent in bottoms_up[1:] :
                if child[1] in parent[1] :
                    parent_child = [ (parent[1], child[1]) ]
                    PClist = [ *PClist, *parent_child ]
                    bottoms_up.remove( child )
                    break
        return ( PClist )

    def linkages_to_graph_dag( self, links:dict ) -> None :
        PClist = self.linkages_to_pclist ( links )
        for pc in PClist :
            self.add_ascendant_descendant ( pc[0], pc[1] )
            self.get_graph()[pc[0]].get_data()['analyte ids'] = [int(a) for a in pc[0].split('.')]
            self.get_graph()[pc[1]].get_data()['analyte ids'] = [int(a) for a in pc[1].split('.')]
        for k,v in links.items():
            self.get_graph()[k].set_metrics([v])
        root_ = self.list_roots()[0]
        self.set_root_id( root_ )

    def distance_matrix_to_pclist ( self , distm:np.array ,
                                    cluster_connections:int = 1 ,
                                    hierarchy_connections:int = 1 ,
                                    bNonRedundant:bool = True  ) -> list :
        #
        # FASTER PCLIST CONSTRUCTION ROUTINE
        # RETURNS LIST USEFUL FOR HIERARCHY GENERATION
        # SHOULD BE EASIER TO PARALLELIZE WITH JIT
        #
        logic = lambda p,c : len(p&c) >= hierarchy_connections and len(p^c)>0
        if not bNonRedundant :
            logic = lambda p,c : len(p&c) >= hierarchy_connections
        #
        R = sorted( list(set( distm.reshape(-1) ) ) )
        prev_clusters = []
        PClist = []
        for r in R :
            present_clusters = self.connectivity ( distm , r , cluster_connections )
            parent_child  = [ (p,c,r) for c in prev_clusters for p in present_clusters \
                          if logic(p,c)  ]
            prev_clusters = present_clusters
            PClist = [ *PClist, *parent_child ]
        return ( PClist )

    def distance_matrix_to_absolute_coordinates ( self , D:np.array , bSquared:bool = False, n_dimensions:int=2 ) -> np.array :
        #
        # SAME AS IN THE IMPETUOUS cluster.py EXCEPT THE RETURN IS TRANSPOSED
        # AND distg.m IN THE RICHTOOLS REPO
        # C++ VERSION HERE https://github.com/richardtjornhammar/RichTools/commit/be0c4dfa8f61915b0701561e39ca906a9a2e0bae
        #
        if not bSquared :
            D = D**2.
        DIM = n_dimensions
        DIJ = D*0.
        M = len(D)
        for i in range(M) :
            for j in range(M) :
                DIJ[i,j] = 0.5* (D[i,-1]+D[j,-1]-D[i,j])
        D = DIJ
        U,S,Vt = np.linalg.svd ( D , full_matrices = True )
        S[DIM:] *= 0.
        Z = np.diag(S**0.5)[:,:DIM]
        xr = np.dot( Z.T,Vt )
        return ( xr.T )

    def calculate_adjacency_matrix( self , bSparse:bool    = False ,
                                    analyte_identifier:str = None  ,
                                    analyte_adjacency_level:int = None  ) -> dict :
        #
        # IF ANALYTE IDENTIFIER IS PASSED THEN CONSTRUCT THE
        # ANALYTE ADJACENCY MATRIX AT A SPECIFIED LEVEL
        # NOTE THAT YOU CAN GET THE ADJACENCY MATRIX FOR ALL
        # ANALYTES VIA : distance_matrix:np.array, level_cutoff:float
        # adj_matrix = distance_matrix<=level_cutoff - np.eye(len(distance_matrix))
        #
        # DEFAULT: CONSTRUCT NODE TO NODE (CLUSTERS) LINK ADJACENCY MATRIX
        #
        def unravel( seq:list )->list :
            if isinstance( seq , (list) ):
                yield from (x for y in seq for x in unravel(y))
            else:
                yield seq

        graph = self.get_graph()
        
        if analyte_identifier is None or analyte_adjacency_level is None :
            names  = list(self.keys())
            Nn     = len(names)
            lookup = {n:i for n,i in zip(names,range(Nn)) }
            if bSparse :
                amat = dict()
            else :
                amat = np.zeros(Nn*Nn).reshape(Nn,Nn)
            for name in names :
                LINKS = []
                for linktype in [ 'links' , 'ascendants' , 'descendants' ]:
                    LINKS.append( graph[name].get_links(linktype) )
                for link in list(unravel(LINKS)):
                    i = lookup[name]
                    j = lookup[link]
                    amat[i,j] = 1
                    amat[j,i] = 1
        else :
            level = analyte_adjacency_level
            root_data = graph[ self.get_root_id() ].get_data()
            if analyte_identifier in root_data :
                names = root_data[ analyte_identifier ]
            else :
                print ( 'ERROR COULD NOT FIND GLOBAL IDENTIFIER INFORMATION:' , analyte_identifier )
                exit (1)
            Nn = len( names )
            nnames = list(self.keys())
            lookup = { a:i for a,i in zip(names,range(Nn)) }
            if bSparse :
                amat = dict()
            else :
                amat = np.zeros(Nn*Nn).reshape(Nn,Nn)
            for name in nnames :
                LINKS = []
                for linktype in [ 'links' , 'ascendants' , 'descendants' ]:
                    LINKS.append( graph[name].get_links(linktype) )
                for link in list(unravel(LINKS)):
                    i_names = graph[ name ].get_data()[ analyte_identifier ]
                    j_names = graph[ link ].get_data()[ analyte_identifier ]
                    if graph[ link ].level()==level or graph[name].level()==level :
                        for namei in i_names :
                            for namej in j_names :
                                i = lookup [ namei ]
                                j = lookup [ namej ]
                                if not i == j :
                                    amat[i,j] = 1
                                    amat[j,i] = 1
        self.adjacency_matrix_ = { 'adjacency matrix':amat , 'index names':names , 'sparsity':bSparse }
        return ( self.adjacency_matrix_ )
    
    def retrieve_adjacency_matrix( self , bForceRecalculate:bool=False ) -> dict :
        if self.adjacency_matrix_ is None or ( not self.adjacency_matrix_ is None and bForceRecalculate ) :
            amat_d = self.calculate_adjacency_matrix()
            self.adjacency_matrix_  = amat_d
        else :
            amat_d = self.adjacency_matrix_
        return ( amat_d )

    def distance_matrix_to_graph_dag ( self , distm:np.array , n_:int=1 , bVerbose:bool=False , names:list=None ) -> None :
        #
        # CONSTRUCTS THE HIERACHY FROM A DISTANCE MATRIX
        # SIMILAR TO THE ROUTINES IN hierarchical.py IN THIS IMPETUOUS REPO
        #
        if len ( distm.shape ) < 2 :
            print ( 'PLEASE SUBMIT A SQUARE DISTANCE MATRIX' )
            exit(1)
        lookup = dict()
        m_ = len(distm)
        for I in range(m_) :
            lookup[I] = I
        if not names is None :
            if len ( names ) == m_ :
                for I,N in zip(range(len(names)),names):
                    lookup[I] = N
        pclist = self.distance_matrix_to_pclist( distm )
        for pc_ in pclist :
            lpc0 = [ lookup[l] for l in list(pc_[0]) ]
            lpc1 = [ lookup[l] for l in list(pc_[1]) ]
            asc = '.'.join([str(l) for l in lpc0])
            des = '.'.join([str(l) for l in lpc1])
            asc_met = pc_[2]
            self.add_ascendant_descendant(asc,des)
            if len( self.get_graph()[asc].get_metrics() ) < 1 :
                self.get_graph()[asc].set_metrics([asc_met])
            self.get_graph()[asc].get_data()['analyte ids'] = lpc0
            self.get_graph()[des].get_data()['analyte ids'] = lpc1
        for key in self.keys() :
            if self.get_graph()[key].is_a_root(n_):
                self.set_root_id ( key )
        if bVerbose :
            self.show()
            print ( self.get_root_id() )
            self.get_graph()[self.get_root_id()].show()

    def graph_analytes_to_approximate_distance_matrix ( self ,
             analyte_identifier:str = 'analyte ids',
             alpha:float = 1. ) -> np.array :

        root_data    = self.get_graph()[ self.get_root_id() ].get_data()

        if analyte_identifier in root_data:
            all_analytes = root_data[ analyte_identifier ]
        else:
            print ( 'ERROR COULD NOT FIND GLOBAL IDENTIFIER INFORMATION:' , analyte_identifier )
            exit (1)
        m_ = len( all_analytes )
        lookup = { a:i for a,i in zip(all_analytes,range(m_)) }
        CM = np.ones(m_*m_).reshape(m_,m_)
        for item in self.get_graph().items() :

            item_data = item[1].get_data()

            if analyte_identifier in item_data : # IMPROVE SPEED HERE
                for q in item_data[analyte_identifier] :
                    for p in item_data[analyte_identifier] : # STOP
                        CM[lookup[p],lookup[q]]+=1
        #
        # CONSTRUCT APPROXIMATION
        # LEVELS MISSING EVEN VIA DEFAULT
        approximate_distm  = 1./CM - np.mean(np.diag(1./CM))
        approximate_distm *= 1-np.eye(m_)
        return ( np.abs(approximate_distm) , lookup )

    def add_ascendant_descendant ( self, ascendant:str, descendant:str ) -> None :
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
        self.add(n)
        self.add(m)

    def generate_ascendants_descendants_lookup ( self ) -> (type(list(str())),type(list(str()))) :
        all_names   = self.keys()
        descendants = [ ( idx , set( self.complete_lineage( idx,linktype='descendants')['path'] ) ) for idx in all_names ]
        ancestors   = [ ( idx , set( self.complete_lineage( idx,linktype='ascendants' )['path'] ) ) for idx in all_names ]
        return ( ancestors , descendants )

    def ascendant_descendant_file_to_dag ( self, relationship_file:str = './PCLIST.txt' ,
                                  i_a:int = 0 , i_d:int = 1 ,
                                  identifier:str = None , sep:str = '\t' ) -> (type(list(str())),type(list(str()))) :

        with open ( relationship_file,'r' ) as input :
            for line in input :
                if not identifier is None :
                    if not identifier in line :
                        continue

                lsp = line.replace('\n','').split( sep )
                ascendant  = lsp[i_a].replace('\n','')
                descendant = lsp[i_d].replace('\n','')

                self.add_ascendant_descendant( ascendant , descendant )

        ancestors , descendants = self.generate_ascendants_descendants_lookup()

        return ( ancestors , descendants )


    def calculate_node_level( self, node:Node , stop_at:str = None , order:str='depth' ) -> None :
        note__ = """
         SEARCHING FOR ASCENDANTS WILL YIELD A
         DIRECT PATH IF A DEPTH SEARCH IS EMPLOYED.
         IF THERE ARE SPLITS ONE MUST BREAK THE SEARCH.
         SPLITS SHOULD NOT BE PRESENT IN ASCENDING DAG
         SEARCHES. SPLIT KILLING IS USED IF depth AND
         stop_at ARE SPECIFIED. THIS CORRESPONDS TO
         DIRECT LINEAGE INSTEAD OF COMPLETE.
        """
        level = len( self.search( root_id=node.identification(), linktype='ascendants', order=order, stop_at=stop_at )['path'] ) - 1
        node.set_level( level )


    def hprint ( self, node:Node, visited:set,
                 I:int = 0, outp:str = "" , linktype:str = "descendants",
                 bCalcLevel = True ) -> (str,int) :
        I = I+1
        if bCalcLevel :
            self.calculate_node_level( node, stop_at = self.get_root_id() )

        head_string   = "{\"source\": \"" + node.identification() + "\", \"id\": " + str(I)
        head_string   = head_string + ", \"level\": " + str(node.level())
        desc_         = str(node.description())
        if len( desc_ ) == 0 :
            desc_ = "\"\""
        head_string   = head_string + ", \"description\": " + desc_
        dat = node.get_data().items()
        if len( dat )>0 :
            for k,v in dat :
                sv = str(v)
                if len(sv) == 0 :
                    sv = "\"\""
                head_string   = head_string + ", \"" + str(k) + "\": " + sv
        met = node.get_metrics()
        head_string = head_string + ", \"metric\": " + str( met[0] if len(met)>0 else 0 )

        desc_h_str    = ", \"children\": ["
        desc_t_str    = "]"
        tail_string   = "}"

        visited = visited|set( [node.identification()] )
        outp    = outp + head_string
        links   = node.get_links(linktype)
        for w in links :
            if not w in visited and len(w)>0 :
                outp = outp + desc_h_str
                outp,I = self.hprint ( self.get_node(w), visited, I, outp, linktype  )
                outp = outp + desc_t_str
        outp = outp + tail_string
        return ( outp,I )

    def rename_data_field_values ( self, lookup:dict = None , field_name:str = 'analyte ids' ) -> None :
        if lookup is None :
            return
        for item in self.items() :
            igdfnl = item[1].get_data()[field_name]
            self.get_graph()[item[0]].get_data()[field_name] =\
                [ n if not n in lookup else lookup[n] for n in igdfnl ]

    def write_json ( self , jsonfile:str = None, bCalcLevel:bool = True ,
                     linktype:str = 'descendants', root_id:str = None ) -> str :
        I:int = 1
        if root_id is None :
            root_id = self.get_root_id()
        v = root_id
        node:Node = self.get_node(v)
        visited   = set()
        json_data_txt,I = self.hprint( node, visited,
                                       linktype   = linktype,
                                       bCalcLevel = bCalcLevel )
        if not jsonfile is None :
            of_ = open(jsonfile,'w')
            print ( json_data_txt,file=of_ )
        return ( json_data_txt )

    def write_gmt ( self, gmtfile:str = None ) -> str :
        gmt_data_txt = "#GROUPNAME\tPARENT:DESC:LVL:MET\tANALYTE1\tANALYTE2\t...\n"
        for item in self.items() :
            asc = ':'.join(item[1].get_links('ascendants'))
            gmt_line = item[0] + '\t' + asc + ':' + str(item[1].description()) + \
                    ':' + str(item[1].level()) + ':' + \
                    ' '.join([str(i) for i in item[1].get_metrics()]) + '\t' + \
                    '\t'.join([str(i) for i in item[1].get_data()['analyte ids']]) + '\n'
            gmt_data_txt = gmt_data_txt + gmt_line

        if not gmtfile is None:
            of_ = open ( gmtfile , 'w' )
            print ( gmt_data_txt , file=of_)
        return ( gmt_data_txt )
    
    def collect_linkages ( self ) -> dict :
        #
        links = dict()
        for item in self.items() :
            if True :
                a_ = 0
                if len( str(item[1].level()) )>0 :
                    a_ = item[1].level()
                mets = item[1].get_metrics()
                if len( mets ) > 0 :
                    a_ = mets[0]
                links[item[0]] = a_
        return ( links )
    
    def write_linkages ( self , linkfile:str=None ) -> str :
        #
        links_ = [ "\"cluster\":"+str(k) + ", \"metric\":" + str(v) for k,v in self.collect_linkages().items() ]
        linkages_txt = '['+']\n['.join(links_)+'] '
        #
        # DEV
        if not linkfile is None:
            of_ = open(linkfile,'w')
            print ( linkages_txt ,file=of_)
        return ( linkages_txt )


class Brain ( NodeGraph ) :
    def __init__ ( self ) :
        # THIS CLASS WILL UTILIZE NEURON SPECIFIC SUBFUNCTIONS
        IQ_:int = 0


def add_attributes_to_node_graph ( p_df:type(pd.DataFrame) , tree:NodeGraph ) -> NodeGraph :
    for idx in p_df.index.values :
        for attribute in p_df.columns.values :
            tree.get_node( idx ).get_data()[attribute] = p_df.loc[idx,attribute]
    return ( tree )


def ascendant_descendant_to_dag ( relationship_file:str = './PCLIST.txt' ,
                                  i_a:int = 0 , i_d:int = 1 ,
                                  identifier:str = None , sep:str = '\t' ) -> NodeGraph :
    RichTree = NodeGraph()
    ancestors , descendants = RichTree.ascendant_descendant_file_to_dag( relationship_file = relationship_file ,
        i_a = i_a , i_d = i_d ,identifier = identifier , sep = sep )

    return ( RichTree , ancestors , descendants )


def drop_duplicate_indices( df ):
    df_ = df.loc[~df.index.duplicated(keep='first')]
    return df_

def write_tree( tree:NodeGraph , outfile='tree.json', bVerbose=True ):
    if bVerbose:
        print ( 'YOU CAN CALL THE NodeGraph METHOD tree.write_json() FUNCTION DIRECTLY' )
    o_json = tree.write_json( outfile )
    return ( o_json )

def add_attributes_to_tree ( p_df , tree ):
    add_attributes_to_node_graph ( p_df , tree )
    return ( tree )

def parent_child_to_dag ( relationship_file:str = './PCLIST.txt' ,
             i_p:int = 0 , i_c:int = 1 , identifier:str = None ) :

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

def normalise_for_apples_and_oranges_stats( X:np.array , method:str='average' ) -> np.array :
    X_ = (rankdata( X , method=method )-0.5)/len(set(X))
    return ( X_ )

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

def read_xyz(fname='argon.xyz',sep=' ') :
    coords = []
    with open(fname,'r') as input:
        for line in input:
            lsp = [u for u in line.replace('\n','').split(sep) if len(u)>0 ]
            print(lsp,len(lsp))
            if len(lsp) == 4:
                coords.append( ( lsp[0],[ float(c) for c in lsp[1:]] ) )
    return ( coords )


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

