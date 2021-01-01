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

import numpy as np
import pandas as pd

def hcurve ( x ) :
    return ( x )

def D ( d , bLinear=False ) :
    if bLinear :
        return ( int(np.floor(3.5*d)) )
    if d==0 :
        return ( 0 )
    else :
        return ( D(d-1) + 2**d  )

def cluster_probability_of_x ( X ,
                    evidence , labels , centroids ,
                    K = None , bLinear = False ) :

    ispanda = lambda P: 'pandas' in str(type(P)).lower()
    BustedPanda = lambda R : R.values if ispanda(R) else R
    
    desc_ = """
    DOING THE SIMPLE PROBABILISTIC MODELLING FOR THE DATA
    YOU CAN USE THIS TO DO VORONOI TESSELATION WITH K = 1 IN 2D
    X         IS A LIST OF COORDINATE POINTS TO EVALUATE THE FUNCTION ON
    EVIDENCE  IS A LIST OF ACTUAL EXPERIMENTAL DATA COORDINATES
    LABELS    ARE THE IMPUTED CLUSTER LABELS OF THE EXPERIMENTAL DATA POINTS
    CENTROIDS ARE THE CENTROID COORDINATES
    K          IS THE NEIGHBOUR DIMENSION
    BLINEAR    SPECIFIES THAT ...

    labels , centroids = impq.knn_clustering_alignment ( P , Q )
    evidence = P.values

    THIS IS NOT THE SAIGA YOU ARE LOOKING FOR -RICHARD TJÖRNHAMMAR 
    """
    ev_ = BustedPanda ( evidence )
    N , DIM = np.shape( ev_ )
    if K is None :
        K = D ( DIM , bLinear )
    #
    # INPUT TYPE CHECKING
    if True :
        bFailure = False
        if not ( 'array' in str(type(labels)) and len(np.shape(labels)) == 2 ) :
            print("THE LABELS SHOULD BE 2D ARRAY OF DIMENSION N,1 "); bFailure=True
        if not ( 'array' in str(type(ev_)) and len(np.shape(ev_)) == 2 ) :
            print("EV_ SHOULD BE 2D ARRAY OF DIMENSION N,DIM "); bFailure=True
        if not ( 'array' in str(type(centroids)) and len(np.shape(centroids)) == 2 ) :
            print("CENTROIDS SHOULD BE 2D ARRAY OF DIMENSION M,DIM "); bFailure=True
        if not ( 'array' in str(type(X)) and len(np.shape(X)) == 2 ) :
            print("X SHOULD BE 2D ARRAY OF DIMENSION P,DIM "); bFailure=True
        
        if bFailure :
            print( desc_ )
            exit(1)
    #        
    all_cluster_names = sorted(list(set( [ l[0] for l in labels] )))
    print ( all_cluster_names )
    #
    Rdf = None
    for x in X :
        S = sorted([ (s,i) for (s,i) in zip( np.sum( (x - ev_)**2 , 1 ) , range(len(ev_)) ) ])[:K]
        present_clusters = [ labels[s[1]][0] for s in S ]
        pY = []
        for yi in all_cluster_names:
            pY.append( np.sum(yi==present_clusters)/K )
        tdf = pd.DataFrame( [[*x,*pY]] ,
                    columns = [ *['d'+str(i) for i in range(len(x))] ,
                                *all_cluster_names ] )
        if Rdf is None :
            Rdf = tdf
        else :
            Rdf = pd.concat([Rdf,tdf])

    Rdf.index = range(len(X))
    probability_df = Rdf.copy()

    return ( probability_df )

import impetuous.quantification as impq
if __name__ == '__main__' :
    #
    # IF YOU REQUIRE THE DATA THEN LOOK IN :
    # https://github.com/richardtjornhammar/RichTools
    # WHERE YOU CAN FIND THE FILES USED HERE
    #
    if True :        
        print( D(1),D(2),D(3),D(4) )
        print( D(1,True),D(2,True),D(3,True),D(4,True) )

        colors = {'H':'#777777','C':'#00FF00','N':'#FF00FF','O':'#FF0000','P':'#FAFAFA'}
        Q = read_xyz( name='data/naj.xyz'   , header=2 , sep=' ' )
        P = read_xyz( name='data/cluster0.xyz' , header=2 , sep='\t' ) # IF IT FAILS THEN CHECK THE FORMATING OF THIS FILE

        if True :
            labels , centroids = impq.knn_clustering_alignment ( P , Q )
            pd.DataFrame(  labels  ) .to_csv( 'labels.csv'   ,'\t' )
            pd.DataFrame( centroids).to_csv( 'centroids.csv','\t' )
        else :
            labels = pd.read_csv('labels.csv','\t',index_col=0).values
            centroids = pd.read_csv('centroids.csv','\t',index_col=0).values

        p = cluster_probability_of_x ( X = np.array( [ [-20.5, 24.6 , 84] , [10,10,10] ] ) ,
                            evidence  = P.values  , 
                            labels    = labels.reshape(-1,1)    ,
                            centroids = centroids ,
                            K = None )
        print ( p )
