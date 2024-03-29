"""
Copyright 2024 RICHARD TJÖRNHAMMAR

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
import typing
import sys
import sklearn.cluster as sc

try :
        from numba import jit
        bUseNumba = True
except ImportError :
        print ( "ImportError:"," NUMBA. WILL NOT USE IT")
        bUseNumba = False
except OSError:
        print ( "OSError:"," NUMBA. WILL NOT USE IT")
        bUseNumba = False

# THE FOLLOWING KMEANS ALGORITHM IS THE AUTHOR OWN LOCAL VERSION
if bUseNumba :
        @jit(nopython=True)
        def seeded_kmeans( dat, cent ):
                #
                # PYTHON ADAPTATION OF MY C++ CODE THAT CAN BE FOUND IN
                # https://github.com/richardtjornhammar/RichTools/blob/master/src/cluster.cc
                # AROUND LINE 2345
                # AGAIN CONSIDER USING THE C++ VERSION SINCE IT IS ALOT FASTER
                # HERE WE SPEED IT UP USING NUMBA IF THE USER HAS IT INSTALLED AS A MODULE
                #
                NN , MM = np.shape ( dat  )
                KK , LL = np.shape ( cent )
                if not LL == MM :
                        print ( 'WARNING DATA FORMAT ERROR. NON COALESCING COORDINATE AXIS' )

                labels = [ int(z) for z in np.zeros(NN) ]
                w = labels
                counts = np.zeros(KK)
                tmp_ce = np.zeros(KK*MM).reshape(KK,MM)
                old_error , error , TOL = 0. , 1. , 1.0E-10
                while abs ( error - old_error ) > TOL :
                        old_error = error
                        error = 0.
                        counts = counts * 0.
                        tmp_ce = tmp_ce * 0.
                        # START BC
                        for h in range ( NN ) :
                                min_distance = 1.0E30
                                for i in range ( KK ) :
                                        distance = np.sum( ( dat[h]-cent[i] )**2 )
                                        if distance < min_distance :
                                                labels[h] = i
                                                min_distance = distance
                                tmp_ce[labels[h]] += dat[ h ]
                                counts[labels[h]] += 1.0
                                error += min_distance
                        # END BC
                        for i in range ( KK ) :
                                if counts[i]>0:
                                        cent[i] = tmp_ce[i]/counts[i]
                centroids = cent
                return ( labels, centroids )
else :
        def seeded_kmeans( dat, cent ):
                #
                # SLOW SLUGGISH KMEANS WITH A DUBBLE FOR LOOP
                # IN PYTHON! WOW! SUCH SPEED!
                #
                NN , MM = np.shape ( dat  )
                KK , LL = np.shape ( cent )
                if not LL == MM :
                        print ( 'WARNING DATA FORMAT ERROR. NON COALESCING COORDINATE AXIS' )

                labels = [ int(z) for z in np.zeros(NN) ]
                w = labels
                counts = np.zeros(KK)
                tmp_ce = np.zeros(KK*MM).reshape(KK,MM)
                old_error , error , TOL = 0. , 1. , 1.0E-10
                while abs ( error - old_error ) > TOL :
                        old_error = error
                        error = 0.
                        counts = counts * 0.
                        tmp_ce = tmp_ce * 0.
                        # START BC
                        for h in range ( NN ) :
                                min_distance = 1.0E30
                                for i in range ( KK ) :
                                        distance = np.sum( ( dat[h]-cent[i] )**2 )
                                        if distance < min_distance :
                                                labels[h] = i
                                                min_distance = distance
                                tmp_ce[labels[h]] += dat[ h ]
                                counts[labels[h]] += 1.0
                                error += min_distance
                        # END BC
                        for i in range ( KK ) :
                                if counts[i]>0:
                                        cent[i] = tmp_ce[i]/counts[i]
                centroids = cent
                return ( labels, centroids )


from scipy.spatial.distance import squareform , pdist
absolute_coordinates_to_distance_matrix = lambda Q:squareform(pdist(Q))

distance_matrix_to_geometry_conversion_notes = """
*) TAKE NOTE THAT THE OLD ALGORITHM CALLED DISTANCE GEOMETRY EXISTS. IT CAN BE EMPLOYED TO ANY DIMENSIONAL DATA. HERE YOU FIND A SVD BASED ANALOG OF THAT OLD METHOD.

*) PDIST REALLY LIKES TO COMPUTE SQUARE ROOT OF THINGS SO WE SQUARE THE RESULT IF IT IS NOT SQUARED.

*) THE DISTANCE MATRIX CONVERSION ROUTINE BACK TO ABSOLUTE COORDINATES USES R2 DISTANCES.
"""

if bUseNumba :
        @jit(nopython=True)
        def distance_matrix_to_absolute_coordinates ( D , bSquared = False, n_dimensions=2 , bLegacy=True ) :
                # C++ https://github.com/richardtjornhammar/RichTools/commit/be0c4dfa8f61915b0701561e39ca906a9a2e0bae
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
                if bLegacy :
                    print ( "YOU ARE RUNNING THE LEGACY IMPLEMENTATION. THE RETURN MATRIX WILL BE CHANGED TO ITS TRANSPOSE" )
                    print ( "TO USE THE NEW VERSION SET bLegacy ARGUMENT TO FALSE" )
                    xr = np.dot( Z.T,Vt )
                    return ( xr )
                else:
                    xr = np.dot( Z.T,Vt )
                    return ( xr.T )
else :
        def distance_matrix_to_absolute_coordinates ( D , bSquared = False, n_dimensions=2 , bLegacy=True ):
                # C++ https://github.com/richardtjornhammar/RichTools/commit/be0c4dfa8f61915b0701561e39ca906a9a2e0bae
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
                if bLegacy :
                    print ( "YOU ARE RUNNING THE LEGACY IMPLEMENTATION. THE RETURN MATRIX WILL BE CHANGED TO ITS TRANSPOSE" )
                    print ( "TO USE THE NEW VERSION SET bLegacy ARGUMENT TO FALSE" )
                    xr = np.dot( Z.T,Vt )
                    return ( xr )
                else:
                    xr = np.tensordot( Z.T,Vt, axes=(0,1) )
                    return ( xr.T )

if bUseNumba :
        @jit(nopython=True)
        def connectivity ( B , val, bVerbose=False ) :
                description = """ This is a cutoff based clustering algorithm. The intended use is to supply a distance matrix and a cutoff value (then becomes symmetric positive).  For a small distance cutoff, you should see all the parts of the system and for a large distance cutoff, you should see the entire system. It has been employed for statistical analysis work as well as the original application where it was employed to segment molecular systems."""
                if bVerbose :
                        print ( "CONNECTIVITY CLUSTERING OF ", np.shape(B), " MATRIX" )
                # PYTHON ADAPTATION OF MY C++ CODE THAT CAN BE FOUND IN
                # https://github.com/richardtjornhammar/RichTools/blob/master/src/cluster.cc
                # AROUND LINE 2277
                # CONSIDER COMPILING AND USING THAT AS A MODULE INSTEAD OF THIS SINCE IT IS
                # A LOT FASTER
                # FOR A DESCRIPTION READ PAGE 30 (16 INTERNAL NUMBERING) of:
                # https://kth.diva-portal.org/smash/get/diva2:748464/FULLTEXT01.pdf
                #
                # https://github.com/richardtjornhammar/RichTools/blob/master/src/cluster.cc
                # ADDED TO RICHTOOLS HERE: https://github.com/richardtjornhammar/RichTools/commit/74b35df9c623bf03570707a24eafe828f461ed90#diff-25a6634263c1b1f6fc4697a04e2b9904ea4b042a89af59dc93ec1f5d44848a26
                # CONNECTIVITY SEARCH FOR (connectivity) CONNECTIVITY
                #
                nr_sq,mr_sq = np.shape(B)
                if nr_sq != mr_sq :
                        print ( 'ERROR: FAILED' )
                N = mr_sq
                res , nvisi, s, NN, ndx, C = [0], [0], [0], [0], [0], 0
                res .append(0)
                for i in range(N) :
                        nvisi.append(i+1)
                        res.append(0); res.append(0)
                        ndx.append(i)

                res   = res[1:]
                nvisi = nvisi[1:]
                ndx   = ndx[1:]
                while ( len(ndx)>0 ) :
                        i = ndx[-1] ; ndx = ndx[:-1]
                        NN = []
                        if ( nvisi[i]>0 ) :
                                C-=1
                                for j in range(N) :
                                        if ( B[i,j]<=val ) :
                                                NN.append(j)
                                while ( len(NN)>0 ) :
                                        # back pop_back
                                        k = NN[-1]; NN = NN[:-1]
                                        nvisi[k] = C
                                        for j in range(N):
                                                if ( B[j,k]<=val ) :
                                                        for q in range(N) :
                                                                if ( nvisi[q] == j+1 ) :
                                                                        NN.append(q)
                if bVerbose : # VERBOSE
                        print ( "INFO "+str(-1*C) +" clusters" )
                Nc = [ 0 for i in range(-1*C) ]
                for q in range(N) :
                        res[  q*2+1 ] = q;
                        res[  q*2   ] = nvisi[q]-C;
                        Nc [res[q*2]]+= 1;
                        if bVerbose :
                                print ( " "+str(res[q*2])+" "+str(res[2*q+1]) )
                if bVerbose:
                        for i in range(-1*C) :
                                print( "CLUSTER "  +str(i)+ " HAS " + str(Nc[i]) + " ELEMENTS")
                return ( Nc , np.array(res[:-1]).reshape(-1,2) )
else :
        def connectivity ( B , val, bVerbose=False ) :
                description="""
This is a cutoff based clustering algorithm. The intended use is to supply a distance matrix and a cutoff value (then becomes symmetric positive).  For a small distanc>
        """
                if bVerbose :
                        print ( "CONNECTIVITY CLUSTERING OF ", np.shape(B), " MATRIX" )
                # PYTHON ADAPTATION OF MY C++ CODE THAT CAN BE FOUND IN
                # https://github.com/richardtjornhammar/RichTools/blob/master/src/cluster.cc
                # AROUND LINE 2277
                # CONSIDER COMPILING AND USING THAT AS A MODULE INSTEAD OF THIS SINCE IT IS
                # A LOT FASTER
                # FOR A DESCRIPTION READ PAGE 30 (16 INTERNAL NUMBERING) of:
                # https://kth.diva-portal.org/smash/get/diva2:748464/FULLTEXT01.pdf
                #
                nr_sq,mr_sq = np.shape(B)
                if nr_sq != mr_sq :
                        print ( 'ERROR' )
                        return ( -1 )
                N = mr_sq
                res , nvisi, s, NN, ndx, C = [], [], [], [], [], 0
                res .append(0)
                for i in range(N) :
                        nvisi.append(i+1)
                        res.append(0); res.append(0)
                        ndx.append(i)
                while ( len(ndx)>0 ) :
                        i = ndx[-1] ; ndx = ndx[:-1]
                        NN = []
                        if ( nvisi[i]>0 ) :
                                C-=1
                                for j in range(N) :
                                        if ( B[i,j]<=val ) :
                                                NN.append(j)
                                while ( len(NN)>0 ) :
                                        # back pop_back
                                        k = NN[-1]; NN = NN[:-1]
                                        nvisi[k] = C
                                        for j in range(N):
                                                if ( B[j,k]<=val ) :
                                                        for q in range(N) :
                                                                if ( nvisi[q] == j+1 ) :
                                                                        NN.append(q)
                if bVerbose : # VERBOSE
                        print ( "INFO "+str(-1*C) +" clusters" )
                Nc = [ 0 for i in range(-1*C) ]
                for q in range(N) :
                        res[  q*2+1 ] = q;
                        res[  q*2   ] = nvisi[q]-C;
                        Nc [res[q*2]]+= 1;
                        if bVerbose :
                                print ( " "+str(res[q*2])+" "+str(res[2*q+1]) )
                if bVerbose:
                        for i in range(-1*C) :
                                print( "CLUSTER "  +str(i)+ " HAS " + str(Nc[i]) + " ELEMENTS")
                return ( Nc , np.array(res[:-1]).reshape(-1,2) )

if bUseNumba :
        @jit(nopython=True)
        def connectedness ( distm:np.array , alpha:float , n_connections:int=1 ) -> list :
            #
            # AN ALTERNATIVE METHOD
            # DOES THE SAME THING AS THE CONNECTIVITY CODE IN MY
            # CLUSTERING MODULE (in src/impetuous/clustering.py )
            # OR IN https://github.com/richardtjornhammar/RichTools/blob/master/src/cluster.cc
            # https://github.com/richardtjornhammar/RichTools/commit/74b35df9c623bf03570707a24eafe828f461ed90#diff-25a6634263c1b1f6fc4697a04e2b9904ea4b042a89af59dc93ec1f5d44848a26
            # CONNECTIVITY SEARCH FOR (connectivity) CONNECTIVITY
            #
            # THIS ROUTINE RETURNS A LIST BELONGING TO THE CLUSTERS
            # WITH THE SET OF INDICES THAT MAPS TO THE CLUSTER
            #
            if len ( distm.shape ) < 2 :
                print ( 'PLEASE SUBMIT A SQUARE DISTANCE MATRIX' )

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
else :
        def connectedness ( distm:np.array , alpha:float , n_connections:int=1 ) -> list :
            #
            # AN ALTERNATIVE METHOD
            # DOES THE SAME THING AS THE CONNECTIVITY CODE IN MY
            # CLUSTERING MODULE (in src/impetuous/clustering.py )
            # OR IN https://github.com/richardtjornhammar/RichTools/blob/master/src/cluster.cc
            # as of commit https://github.com/richardtjornhammar/RichTools/commit/76201bb07687017ae16a4e57cb1ed9fd8c394f18 2016
            # CONNECTIVITY SEARCH FOR (connectivity) CONNECTIVITY
            #
            # THIS ROUTINE RETURNS A LIST BELONGING TO THE CLUSTERS
            # WITH THE SET OF INDICES THAT MAPS TO THE CLUSTER
            #
            if len ( distm.shape ) < 2 :
                print ( 'PLEASE SUBMIT A SQUARE DISTANCE MATRIX' )

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

if bUseNumba :
        @jit(nopython=True)
        def connectivity_boolean ( B , bVerbose=False ) :
                description = """ This is a cutoff based clustering algorithm. The intended use is to supply a distance matrix and a cutoff value (then becom>
m. It has been employed for statistical analysis work as well as the original application where it was employed to segment molecular systems."""
                if bVerbose :
                        print ( "CONNECTIVITY CLUSTERING OF ", np.shape(B), " MATRIX" )
                # PYTHON ADAPTATION OF MY C++ CODE THAT CAN BE FOUND IN
                # https://github.com/richardtjornhammar/RichTools/blob/master/src/cluster.cc
                # AROUND LINE 2277
                # CONSIDER COMPILING AND USING THAT AS A MODULE INSTEAD OF THIS SINCE IT IS
                # A LOT FASTER
                # FOR A DESCRIPTION READ PAGE 30 (16 INTERNAL NUMBERING) of:
                # https://kth.diva-portal.org/smash/get/diva2:748464/FULLTEXT01.pdf
                #
                # https://github.com/richardtjornhammar/RichTools/blob/master/src/cluster.cc
                # ADDED TO RICHTOOLS HERE: https://github.com/richardtjornhammar/RichTools/commit/74b35df9c623bf03570707a24eafe828f461ed90#diff-25a6634263c1b>
                # CONNECTIVITY SEARCH FOR (connectivity) CONNECTIVITY
                #
                nr_sq,mr_sq = np.shape(B)
                if nr_sq != mr_sq :
                        print ( 'ERROR: FAILED' )
                N = mr_sq
                res , nvisi, s, NN, ndx, C = [0], [0], [0], [0], [0], 0
                res .append(0)
                for i in range(N) :
                        nvisi.append(i+1)
                        res.append(0); res.append(0)
                        ndx.append(i)

                res   = res[1:]
                nvisi = nvisi[1:]
                ndx   = ndx[1:]
                while ( len(ndx)>0 ) :
                        i = ndx[-1] ; ndx = ndx[:-1]
                        NN = []
                        if ( nvisi[i]>0 ) :
                                C-=1
                                for j in range(N) :
                                        if ( B[i,j] ) :
                                                NN.append(j)
                                while ( len(NN)>0 ) :
                                        # back pop_back
                                        k = NN[-1]; NN = NN[:-1]
                                        nvisi[k] = C
                                        for j in range(N):
                                                if ( B[j,k] ) :
                                                        for q in range(N) :
                                                                if ( nvisi[q] == j+1 ) :
                                                                        NN.append(q)
                if bVerbose : # VERBOSE
                        print ( "INFO "+str(-1*C) +" clusters" )
                Nc = [ 0 for i in range(-1*C) ]
                for q in range(N) :
                        res[  q*2+1 ] = q;
                        res[  q*2   ] = nvisi[q]-C;
                        Nc [res[q*2]]+= 1;
                        if bVerbose :
                                print ( " "+str(res[q*2])+" "+str(res[2*q+1]) )
                if bVerbose:
                        for i in range(-1*C) :
                                print( "CLUSTER "  +str(i)+ " HAS " + str(Nc[i]) + " ELEMENTS")
                return ( Nc , np.array(res[:-1]).reshape(-1,2) )
else:
        def connectivity_boolean ( B , bVerbose=False ) :
                description = """ This is a cutoff based clustering algorithm. The intended use is to supply a distance matrix and a cutoff value (then becom>
m. It has been employed for statistical analysis work as well as the original application where it was employed to segment molecular systems."""
                if bVerbose :
                        print ( "CONNECTIVITY CLUSTERING OF ", np.shape(B), " MATRIX" )
                # PYTHON ADAPTATION OF MY C++ CODE THAT CAN BE FOUND IN
                # https://github.com/richardtjornhammar/RichTools/blob/master/src/cluster.cc
                # AROUND LINE 2277
                # CONSIDER COMPILING AND USING THAT AS A MODULE INSTEAD OF THIS SINCE IT IS
                # A LOT FASTER
                # FOR A DESCRIPTION READ PAGE 30 (16 INTERNAL NUMBERING) of:
                # https://kth.diva-portal.org/smash/get/diva2:748464/FULLTEXT01.pdf
                #
                # https://github.com/richardtjornhammar/RichTools/blob/master/src/cluster.cc
                # ADDED TO RICHTOOLS HERE: https://github.com/richardtjornhammar/RichTools/commit/74b35df9c623bf03570707a24eafe828f461ed90#diff-25a6634263c1b>
                # CONNECTIVITY SEARCH FOR (connectivity) CONNECTIVITY
                #
                nr_sq,mr_sq = np.shape(B)
                if nr_sq != mr_sq :
                        print ( 'ERROR: FAILED' )
                N = mr_sq
                res , nvisi, s, NN, ndx, C = [0], [0], [0], [0], [0], 0
                res .append(0)
                for i in range(N) :
                        nvisi.append(i+1)
                        res.append(0); res.append(0)
                        ndx.append(i)

                res   = res[1:]
                nvisi = nvisi[1:]
                ndx   = ndx[1:]
                while ( len(ndx)>0 ) :
                        i = ndx[-1] ; ndx = ndx[:-1]
                        NN = []
                        if ( nvisi[i]>0 ) :
                                C-=1
                                for j in range(N) :
                                        if ( B[i,j] ) :
                                                NN.append(j)
                                while ( len(NN)>0 ) :
                                        # back pop_back
                                        k = NN[-1]; NN = NN[:-1]
                                        nvisi[k] = C
                                        for j in range(N):
                                                if ( B[j,k] ) :
                                                        for q in range(N) :
                                                                if ( nvisi[q] == j+1 ) :
                                                                        NN.append(q)
                if bVerbose : # VERBOSE
                        print ( "INFO "+str(-1*C) +" clusters" )
                Nc = [ 0 for i in range(-1*C) ]
                for q in range(N) :
                        res[  q*2+1 ] = q;
                        res[  q*2   ] = nvisi[q]-C;
                        Nc [res[q*2]]+= 1;
                        if bVerbose :
                                print ( " "+str(res[q*2])+" "+str(res[2*q+1]) )
                if bVerbose:
                        for i in range(-1*C) :
                                print( "CLUSTER "  +str(i)+ " HAS " + str(Nc[i]) + " ELEMENTS")
                return ( Nc , np.array(res[:-1]).reshape(-1,2) )


def clustering_bounds ( distm:np.array ) -> tuple :
    #
    # RETURN THE BOUNDS FOR WHEN THE CLUSTERING SOLUTION IS
    # DISCONNECTED CONTRA FULLY CONNECTED
    #
    d = distm.copy()
    ma = np.min(np.max(d,0))
    for i in range(len(d)):
        d[i,i] = np.inf
    return ( np.min(d) , ma )


def pidx2lidx ( n:int,i:int,j:int )->int :
    from scipy.special import binom
    p = i if i>j else j
    i = i if i<j else j
    j = p
    return ( int ( binom(n,2) - binom(n-i,2) + ( j-i-1 ) )  )

def linear2coord ( N:int, bType=0 ) -> dict :
    if bType == 0 :
        return ( { pidx2lidx(N,i,j):(i,j)  for i in range(N) for j in range(N) if j>i } )
    from scipy.spatial.distance import squareform
    tmpsq =  squareform ( [ i for i in range(int( N*(N-1)*0.5 )) ] )
    return (  { tmpsq[i,j]:(i,j) for i in range(N) for j in range(N) if j>i } )

if True :
    #@njit #@jit( nopython=True )
    def link_clustering_core ( pdp:np.array , N:int ) :
        pdp  = sorted(pdp)[::-1]
        idxs = [ tuple([n]) for n in range(N) ]
        ids0 = idxs.copy()
        cluster_ledger = [ [ 0, idxs ] ]
        for i in range(len(pdp)) :
            p_ = pdp.pop()
            m_ = set( p_[1] )
            for j_ in range(len(idxs)) :
                s_ = idxs.pop()
                if len(m_&set(s_))>0:
                    m_ = m_|set(s_)
                else :
                    idxs = [ s_,*idxs ]
            idxs = [ tuple(sorted(m_)) ,*idxs ]
            if set(idxs) == set(ids0) :
                continue
            cluster_ledger.append( [ p_[0] , idxs.copy() ] )
            ids0 = idxs
            if len(idxs) == 1 : # WE ARE FULLY CONNECTED
                break
        return ( cluster_ledger )

def link_clustering ( input:np.array ) :
    from scipy.spatial.distance import pdist,squareform
    import time,gc
    #
    # INPUT CHECK
    #
    nm = np.shape(input)
    if len(nm) == 1 : # ASSUME PDIST ARRAY
        pdc = input.copy()
        distm = squareform( pdc )
    else :
        if nm[0] == nm[1] : # ASSUME DISTANCE MATRIX
            distm = input.copy()
            pdc = squareform( input )
        else : # ASSUME COORDINATES
            pdc     = pdist ( input )
            distm = squareform ( pdc )
    nm = np.shape(distm)
    N  = nm[0]
    lookup  = linear2coord( N )
    fully_connected_at = np.min(np.max( distm ,0 ))
    del distm
    gc.collect()
    pdp     = [ (d,lookup[i] ) for d,i in zip(pdc,range(len(pdc))) if d <= fully_connected_at ]
    del lookup
    gc.collect()
    del pdc
    gc.collect()
    results = link_clustering_core( pdp , N )
    return ( results )


def connectivity_clustering( distm:np.array, cutoff:float, bBool:bool=False ) -> tuple :
    if bBool :
        # THIS WILL BE FASTER AND MORE MEMORY EFFICIENT
        # IF THE DISTM IS LARGE
        distm_b = distm<=cutoff
        res = connectivity_boolean( distm_b )
    else :
        res = connectivity( distm , cutoff )
    return ( res )


clustering_algorithm = None
clustering_algorithm = sc.KMeans(10) # CHOOSE SOMETHING YOU LIKE NOT THIS

class Cluster(object):
    def __init__( self, nbins=50, nclusters=-1 , use_ranks = False ) :
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from numpy import histogram2d
        from scipy.stats import rankdata
        self.use_ranks = use_ranks
        self.nclusters = nclusters
        self.nbins = nbins
        self.histogram2d = histogram2d
        self.KMeans = KMeans
        self.rankdata = rankdata
        self.pca_f = PCA(2)
        self.centroids_ = None
        self.labels_ = None
        self.df_ = None
        self.num_index_ = None
        self.components_ = None

    def approximate_density_clustering( self, df, nbins=None ) :
        #
        # GENES APPROX 20K OK SO APPROX 50 BINS
        # ANALYTES ON ROWS, SAMPLE POINTS ON COLUMNS
        if nbins is None :
            nbins = self.nbins
        self.df_= df
        frac_df = df
        if self.use_ranks :
            frac_df .apply( lambda x:self.rankdata( x , method='average' )/float(len(x)) )
        self.pca_f.fit(frac_df.T.values)
        self.components_ = self.pca_f.components_
        vals,xe,ye = self.histogram2d(self.pca_f.components_[0],self.pca_f.components_[1],bins=nbins)
        mvs, svsx, svsy = np.mean(vals),np.std(vals,0),np.std(vals,1)
        svs = np.sqrt(svsx**2+svsy**2)
        #
        # IS THERE A DENSITY PEAK SEPARABLE FROM THE MEAN
        # SHOULD DO GRADIENT REJECTION BASED ON TTEST PVALUES
        hits = vals>mvs+0.5*svs
        #
        xe_,ye_ = 0.5*(xe[:1]+xe[1:]) , 0.5*(ye[:1]+ye[1:])
        idx = np.where(hits); xi,yj = idx[0],idx[1]
        centroids = [ (xe[ri],ye[rj]) for (ri,rj) in zip(xi,yj) ]
        if self.nclusters == -1 :
            self.nclusters = len ( centroids )
        if self.nclusters  < len ( centroids ) :
            import heapq
            from scipy.spatial import distance as distance_
            a = distance_.cdist ( centroids, centroids, 'euclidean' )
            cent_idx = heapq.nlargest ( self.nclusters, range(len(a)), a.reshape(-1).__getitem__ )
            centroids = [ centroids[ idx ] for idx in cent_idx ]

        kmeans = self.KMeans(len(centroids),init=np.array(centroids))
        kmeans.fit(self.pca_f.components_.T)
        centers = np.array(kmeans.cluster_centers_).T
        self.labels_ = kmeans.labels_
        self.centroids_ = centers
        self.analyte_dict_ = { c:[] for c in self.labels_ }
        [self.analyte_dict_[self.labels_[i]].append(df.index[i]) for i in range(len(self.labels_)) ]
        return ( self.analyte_dict_ )

    def write_gmt(self, filename = './cluster_file.gmt' ) :
        with open(filename,'w') as of :
            for k,v in self.analyte_dict_.items() :
                print ( 'CLU-'+str(k),'\tDESCRIPTION\t'+'\t'.join(v), file=of )


class ManifoldClustering ( Cluster ) :
    def __init__( self , nbins=50 ) :
        from sklearn.cluster import KMeans
        from sklearn.manifold import MDS, TSNE
        from numpy import histogram2d
        from scipy.stats import rankdata
        self.nbins = nbins
        self.histogram2d = histogram2d
        self.KMeans = KMeans
        self.rankdata = rankdata
        self.mds  = MDS ( n_components=2 )
        self.tsne = TSNE ( n_components=2 )
        self.man = None
        self.centroids_ = None
        self.labels_ = None
        self.df_ = None
        self.num_index_ = None
        self.components_ = None

    def approximate_embedding( self, df, nbins=None , use_tsne=True ) :
        self.man = self.tsne
        if not use_tsne :
            self.man = self.mds
            print ( 'WARNING::SLOW AND WASTEFUL' )
        if nbins is None :
            nbins = self.nbins
        self.df_= df
        frac_df = df.apply( lambda x:self.rankdata( x , method='average' )/float(len(x)) )
        self.components_ = np.array(self.man.fit_transform(frac_df.values)).T
        vals,xe,ye = self.histogram2d(self.components_[0],self.components_[1],bins=nbins)
        mvs, svsx, svsy = np.mean(vals),np.std(vals,0),np.std(vals,1)
        svs = np.sqrt( svsx**2 + svsy**2 )
        #
        # IS THERE A DENSITY PEAK SEPARABLE FROM THE MEAN
        # SHOULD DO GRADIENT REJECTION BASED ON TTEST PVALUES
        hits = vals>mvs+0.5*svs
        #print(hits,vals)
        xe_,ye_=0.5*(xe[:1]+xe[1:]),0.5*(ye[:1]+ye[1:])
        idx = np.where(hits); xi,yj = idx[0],idx[1]
        centroids = [ (xe[ri],ye[rj]) for (ri,rj) in zip(xi,yj) ]
        #
        kmeans = self.KMeans(len(centroids),init=np.array(centroids))
        kmeans.fit(self.components_.T)
        centers = np.array(kmeans.cluster_centers_).T
        self.labels_ = kmeans.labels_
        self.centroids_ = centers
        self.analyte_dict_ = { c:[] for c in self.labels_ }
        [self.analyte_dict_[self.labels_[i]].append(df.index[i]) for i in range(len(self.labels_)) ]
        return ( self.analyte_dict_ )


def run_clustering_and_write_gmt( df , ca , filename = './approx_cluster_file.gmt' ) :
    labels = ca.fit_predict(df.values)
    llabs = [ l for l in labels ]; ulabs=set(llabs)
    with open(filename,'w') as of :
        for ulab in ulabs :
            analytes = df.iloc[llabs==ulab].index.values
            print ( 'CLU-'+str(ulab),'\tDESCRIPTION\t'+'\t'.join(analytes), file=of )


def projection_knn_assignment ( projected_coords , df , NMaxGuess=-1 , n_dimensions=2  ) :
    coords_s = projected_coords.dropna( 0 )
    centroid_coordinates = []
    for row in df.T :
        guess = sorted ( [ (v,i) for (v,i) in zip( df.loc[row].values,df.loc[row].index ) ] ) [::-1][:NMaxGuess]
        maxWeights = [ i[1] for i in guess ]
        use = df.loc[row,maxWeights]
        S = np.sum ( use.values )
        S = 1. if S==0 else S
        crd = np.dot(use.values,coords_s.loc[use.index.values].values)/S
        centroid_coordinates.append(crd)

    centroids_df = pd.DataFrame ( centroid_coordinates , index=df.index , columns=[ 'C'+str(i) for i in range(n_dimensions) ] )
    labels , centroids = seeded_kmeans( coords_s.values,centroids_df.values )
    coords_s.loc[:,'owner'] = centroids_df.iloc[labels].index.values
    for i in range(len(centroids.T)) :
        centroids_df.loc[:,'E'+str(i) ] = (centroids.T)[i]
    return ( centroids_df , coords_s )


def make_clustering_visualisation_df ( CLUSTER , df=None , add_synonyms = False ,
                                    output_name = 'feature_clusters_output.csv' 
                                  ) :
    x_pc1 = CLUSTER.components_[0]
    y_pc2 = CLUSTER.components_[1]
    L_C   = len(CLUSTER.centroids_[0])
    #
    # MAKE CLUSTER COLORS
    make_hex_colors = lambda c : '#%02x%02x%02x' % (c[0]%256,c[1]%256,c[2]%256)
    C0 = [255,255,255] ; cluster_colors = []
    #
    for i in CLUSTER.labels_ :
        C0_ = C0 ; C0_[i%3] = int(np.floor(C0[i%3]-(i/float(L_C))*255))
        cluster_colors.append(make_hex_colors(C0_))

    if not df is None :
        if add_synonyms :
            synonyms = [ ens2sym[df.index.values[i]][0] if df.index.values[i] in ens2sym \
                else ens2sym_2[df.index.values[i]] if df.index.values[i] in ens2sym_2 \
                else df.index.values[i] for i in range(len(px))]
        else :
            synonyms = df.index.values
    data = []
    for (x,y,t,cl,co) in zip( x_pc1,y_pc2,synonyms , [cl for cl in CLUSTER.labels_] ,
                              [cluster_colors[cl] for cl in CLUSTER.labels_] ) :
        data.append([x,y,t,cl,co])
    clustering_df = pd.DataFrame( data , columns = ['X','Y','Type','Cluster','Color'])
    if not df is None :
        clustering_df.index =  df.index.values 
    clustering_df.to_csv( output_name , '\t' )
    return ( clustering_df )

def backprojection_clustering ( analyte_df , bRanked=False , n_dimensions=2 ,
                                bDoFeatures=True , bDoSamples=True ) :
    from scipy.stats import rankdata
    if bRanked :
        rana_df = analyte_df .apply( lambda x:(rankdata(x,'average')-0.5)/len(x) )
    else :
        rana_df = analyte_df

    dimcrdnames = [ 'd'+str(i) for i in range(n_dimensions) ]
    #
    # Do backprojection clustering
    cluster_coords_f = None
    if bDoFeatures :
        #
        dM1 = absolute_coordinates_to_distance_matrix( rana_df.values   )
        #pd.DataFrame(dM1,index=rana_df.index,columns=rana_df.index).to_csv('../data/dM1.tsv','\t')
        #
        # Project it back onto first two components
        max_var_projection = distance_matrix_to_absolute_coordinates ( dM1 , n_dimensions=n_dimensions )
        cluster_coords_f = pd.DataFrame( max_var_projection ,
                                    columns = rana_df.index ,
                                    index = dimcrdnames ).T
    cluster_coords_s = None
    if bDoSamples :
        #
        # And again for all the samples
        dM2 = absolute_coordinates_to_distance_matrix( rana_df.T.values )
        #pd.DataFrame(dM2,index=rana_df.columns,columns=rana_df.columns).to_csv('../data/dM2.tsv','\t')
        #
        # This algorithm is exact but scales somewhere between n^2 and n log n
        max_var_projection = distance_matrix_to_absolute_coordinates ( dM2 , n_dimensions=n_dimensions )
        cluster_coords_s = pd.DataFrame( max_var_projection ,
                                    columns = rana_df.columns ,
                                    index = dimcrdnames ).T
        #cluster_coords_s.to_csv('../data/conclust_s.tsv','\t')

    return ( cluster_coords_f,cluster_coords_s )

def dbscan ( data_frame = None , distance_matrix = None ,
        eps = None, minPts = None , bVerbose = False ) :
    if bVerbose :
        print ( "THIS IMPLEMENTATION FOR DBSCAN" )
        print ( "ASSESSMENT OF NOISE DIFFERS FROM" )
        print ( "THE IMPLEMENTATION FOUND IN SKLEARN")
    #
    # FOR A DESCRIPTION OF THE CONNECTIVITY READ PAGE 30 (16 INTERNAL NUMBERING) of:
    # https://kth.diva-portal.org/smash/get/diva2:748464/FULLTEXT01.pdf
    #from impetuous.clustering import absolute_coordinates_to_distance_matrix
    #from impetuous.clustering import connectivity
    import operator
    if not operator.xor( data_frame is None , distance_matrix is None ) :
        print ( "ONLY SUPPLY A SINGE DATA FRAME OR A DISTANCE MATRIX" )
        print ( "dbscan FAILED" )
        print ( "DATA MATRICES NEEDS TO BE SPECIFIED WITH \" distance_matrix = ... \" " )
        exit(1)
    if not data_frame is None :
        if not 'pandas' in str(type(data_frame)) :
            print ( "ONLY SUPPLY A SINGE DATA FRAME WITH ABSOLUTE COORDINATES" )
            print ( "DATA MATRICES NEEDS TO BE SPECIFIED WITH \" data_frame = ... \" " )
            print ( "dbscan FAILED" )
            exit ( 1 )
        if bVerbose :
            print ( data_frame )
        distance_matrix = absolute_coordinates_to_distance_matrix(data_frame.values)
    if not ( 'float' in str(type(eps)).lower() and 'int' in str(type(minPts)).lower() ) :
        print ( "TO CALL THE dbscan PLEASE SPECIFY AT LEAST A DATA FRAME OR")
        print ( "ITS CORRESPONDING DISTANCE MATRIX AS WELL AS THE DISTANCE CUTOFF PARAMETER" )
        print ( "AND THE MINIMAL AMOUNT OF NEIGHBOUR POINTS TO CONSIDER IT CLUSTERED")
        print ( "dbscan ( data_frame = None , distance_matrix = None , eps = None, minPts = None )" )
    if 'panda' in str(type(distance_matrix)).lower() :
        distance_matrix = distance_matrix.values
    distance_matrix_ = distance_matrix.copy()
    isNoise = np.sum(distance_matrix_<eps,0)-1 < minPts
    i_ = 0
    for ib in isNoise :
        if ib :
            distance_matrix_ [ i_] = ( 1+eps )*10.0
            distance_matrix_.T[i_] = ( 1+eps )*10.0
            distance_matrix_[i_][i_] = 0.
        i_ = i_+1
    clustercontent , clustercontacts  =  connectivity(distance_matrix_,eps)
    return ( {'cluster content': clustercontent, 'clusterid-particleid' : clustercontacts, 'is noise':isNoise} )

def reformat_dbscan_results ( results ) :
    if True :
        clusters = {}
        for icontent in range(len(results['cluster content'])) :
            content = results[ 'cluster content' ][ icontent ]
            for c in results [ 'clusterid-particleid' ] :
                if c[0] == icontent :
                    if results[ 'is noise' ][c[1]] :
                        icontent=-1
                    if icontent in clusters:
                        clusters[ icontent ] .append( c[1] )
                    else :
                        clusters[ icontent ] = [ c[1] ]
        return ( clusters )


if bUseNumba :
    @jit(nopython=True)
    def exclusive_pdist ( P , Q ) :
        Np , Nq = len(P), len(Q)
        R2 = np.zeros(Np*Nq).reshape(Np,Nq)
        for i in range(len(P)):
            for j in range(len(Q)):
                R2[i][j] = np.sum((P[i]-Q[j])**2)
        return ( R2 )
else :
    def exclusive_pdist ( P , Q ) :
        Np , Nq = len(P), len(Q)
        R2 = np.zeros(Np*Nq).reshape(Np,Nq)
        for i in range(len(P)):
            for j in range(len(Q)):
                R2[i][j] = np.sum((P[i]-Q[j])**2)
        return ( R2 )

def select_from_distance_matrix(boolean_list,distance_matrix):
    return ( np.array( [ d[boolean_list] for d in distance_matrix[boolean_list]] ) )

def diar ( n ):
    if n>1:
        return ( np.sqrt(n)*diar(n-1) )
    else:
        return ( 1. )

def calculate_rdf ( particles_i = None , particles_o = None , nbins=100 ,
                    distance_matrix = None , bInGroup = None , bNotInGroup = None ,
                    n_dimensions = 3 , xformat="%.3f" ,
                    constant=4.0/3.0 , rho=1.0 , rmax=None ,
                    bRemoveZeros = False ) :

    import operator
    crit0 = particles_i is None
    crit1 = particles_i is None and particles_o is None
    crit2 = bInGroup is None and distance_matrix is None and bNotInGroup is None

    if not crit2 :
        particles_i = distance_matrix_to_absolute_coordinates ( \
                         select_from_distance_matrix ( bInGroup    , distance_matrix ) ,
                         n_dimensions = n_dimensions ).T
        particles_o = distance_matrix_to_absolute_coordinates ( \
                         select_from_distance_matrix ( bNotInGroup , distance_matrix ) ,
                         n_dimensions = n_dimensions ).T

    if operator.xor( (not crit1) or (not crit0)  , not crit2 ) :
        if not crit0 and particles_o is None :
            particles_o = particles_i
            bRemoveZeros = True
        rdf_p = pd.DataFrame ( exclusive_pdist ( particles_i , particles_o ) ).apply( np.sqrt ).values.reshape(-1)
        if bRemoveZeros :
            rdf_p = [ r for r in rdf_p if not r==0. ]
        if rmax is None :
            rmax  = np.max ( rdf_p ) / diar( n_dimensions+1 )

        rdf_p  = np.array ( [ r for r in rdf_p if r < rmax ] )
        Y_ , X = np.histogram ( rdf_p , bins=nbins )
        X_     = 0.5 * ( X[1:]+X[:-1] )

        norm   = constant * np.pi * ( ( X_ + np.diff(X) )**(n_dimensions) - X_**(n_dimensions) ) * rho
        dd     = Y_ / norm
        rd     = X_

        rdf_source = {'density_values': dd, 'density_ids':[ xformat % (d) for d in rd ] }
        return ( rdf_source , rdf_p )
    else :
        print ( """calculate_rdf ( particles_i = None , particles_o = None , nbins=100 ,
                    distance_matrix = None , bInGroup = None , bNotInGroup = None ,
                    n_dimensions = 3 , xformat="%.3f" ,
                    constant=4.0/3.0 , rho=1.0 , rmax=None ,
                    bRemoveZeros = False )""")
        exit ( 1 )

def unpack ( seq ) : # seq:Union -> Union
    if isinstance ( seq,(list,tuple,set)) :
        yield from ( x for y in seq for x in unpack(y) )
    elif isinstance ( seq , dict ):
        yield from ( x for item in seq.items() for y in item for x in unpack(y) )
    else :
        yield seq

def rem ( a:list , H:list ) -> list :
    h0 = []
    for h in H:
        hp = h - np.sum(h>np.array(h0))
        h0 .append(h)
        a .pop(hp)
    return(a)

def nppop(A:np.array, irow:int=None, jcol:int=None ) -> list[np.array] :
    # ASSUMES ROW MAJOR ORDER
    rrow:np.array() = None
    rcol:np.array() = None
    N = len(A)
    M0,M1 = np.shape(A)
    if not irow is None :
        rrow = A[irow,:]
        A    = np.delete(A,range(N*irow,N*(irow+1))).reshape(-1,N)
        M0   = M0-1
    if not jcol is None :
        rcol = A[:,jcol]
        A    = np.delete(A,range(jcol,len(A.reshape(-1)),N) )
        M1   = M1-1
    return ( [rrow,rcol,A.reshape(M0,M1)] )

def link1_ ( D:np.array , method:str = 'min' , bDA:bool = False ) -> list :
    def func( r:float , c:float , lab:str='min' ) -> float :
        if lab == 'max' :
            return ( r if r > c else c )
        if lab == 'min' :
            return ( r if r < c else c )
    #
    nmind = np.argmin(D) # SIMPLE TIEBREAKER
    if bDA :
        planar_crds = lambda linear_crd,N : tuple( (int(linear_crd/N) , linear_crd%N) )
        #
        # HEURISTIC TIEBREAKER
        dr    = D.reshape(-1)
        ties  = np.where(dr==dr[nmind])[0]
        ties  = ties[:int(len(ties)/2)]
        if len(ties) > 1 :
            nmind = ties[np.argmin( [ np.sum( D[planar_crds(t,len(D))[0],:]) for t in ties ]) ]
    ( i,j )  = ( int(nmind/len(D)) , nmind%len(D) )
    k = j - int(i<j)
    l = i - int(j<i)
    pop1 = nppop(D,i,j)
    pop2 = nppop(pop1[-1],k,l)
    lpr  = list(pop2[0])
    d    = lpr.pop(l)
    lpr  = np.array(lpr)
    lpc  = pop2[1]
    nvec = np.array([*[D[0,0]],*[ func(r,c,method) for (r,c) in zip(lpr,lpc) ]])
    DUM  = np.eye(len(nvec))*0
    DUM[ 0  , : ] = nvec
    DUM[ :  , 0 ] = nvec
    DUM[ 1: , 1:] = pop2[-1]
    return ( [ DUM , (i,j) , d ]  )

def linkage_dict_tuples ( D:np.array , method:str = 'min' ) -> dict :
    N   = len(D)
    dm  = np.max(D)*1.1
    idx = list()
    for i in range(N): D[i,i] = dm; idx.append(i)
    cidx     = []
    sidx     = set()
    res      = [D]
    linkages = dict()
    while ( len(res[0]) > 1 ) :
        res   = link1_ ( res[0] , method )
        oidx  = tuple ( unpack( tuple( [ idx[i] for i in res[1] ]) ) )
        unique_local_clusters = [ c for c in cidx if len( set(c) - set(oidx) ) >0 ]
        unique_local_clusters .append( oidx )
        cidx  .append( oidx )
        sidx  = sidx|set(oidx)
        idx   = [*unique_local_clusters[::-1] , *[i for i in range(N) if not i in sidx ]]
        linkages[ oidx ] = res[-1]
    for i in range(N) :
        linkages[ (i,) ] = 0
    return ( linkages )

def link0_ ( D:np.array , method:str = 'min' ) -> list :
    def func( r:float , c:float , lab:str='min' ) -> float :
        if lab == 'max' :
            return ( r if r > c else c )
        if lab == 'min' :
            return ( r if r < c else c )

    nmind = np.argmin(D) # SIMPLE TIEBREAKER
    ( i,j )  = ( int(nmind/len(D)) , nmind%len(D) )
    k = j - int(i<j)
    l = i - int(j<i)

    pop1 = nppop(D,i,j)
    pop2 = nppop(pop1[-1],k,l)
    lpr  = list(pop2[0])
    d    = lpr.pop(l)
    lpr  = np.array(lpr)
    lpc  = pop2[1]
    nvec = np.array([*[D[0,0]],*[ func(r,c,method) for (r,c) in zip(lpr,lpc) ]])
    DUM  = np.eye(len(nvec))*0
    DUM[ 0  , : ] = nvec
    DUM[ :  , 0 ] = nvec
    DUM[ 1: , 1:] = pop2[-1]
    return ( [ DUM , (i,j) , d ]  )

def linkages_tiers ( D:np.array , method:str = 'min' ) -> dict :
    N   = len(D)
    dm  = np.max(D)*1.1
    idx = list()
    for i in range(N):  D[i,i] = dm ; idx.append( tuple((i,)) )
    cidx     = []
    sidx     = set()
    res      = [D]
    linkages = dict()
    while ( len(res[0]) > 1 ) :
        res          = link0_ ( res[0] , method )
        found_cidx   = tuple( [ idx[i] for i in res[1] ])
        idx = [ *[found_cidx], *[ix_ for ix_ in idx if not ix_ in set(found_cidx) ] ]
        linkages[ found_cidx ] = res[-1]
    for i in range(N) :
        linkages[ (i,) ] = 0
        D[i,i] = 0
    return ( linkages )

def lint2lstr ( seq:list[int] ) -> list[str] :
    #
    # DUPLICATED IN special.lint2lstr
    if isinstance ( seq,(list,tuple,set)) :
        yield from ( str(x) for y in seq for x in lint2lstr(y) )
    else :
        yield seq

def sclinkages ( distm ,command='min' , bStrKeys=True ) -> dict :
    #
    # FROM impetuous.clustering
    # RETURN TYPE DIFFERS
    #
    from scipy.spatial.distance  import squareform
    from scipy.cluster.hierarchy import linkage as sclinks
    from scipy.cluster.hierarchy import fcluster
    cmd = command
    if command in set(['min','max']) :
        cmd = {'min':'single','max':'complete'}[command]
    Z  = sclinks( squareform(distm) , cmd )
    CL = {}
    F  = {} # NEW
    for d in Z[:,2] :
        row  = fcluster ( Z ,d, 'distance' )
        F[d] = row # NEW
        sv_ = sorted(list(set(row)))
        cl  = {s:[] for s in sv_}
        for i in range( len( row ) ) :
            cl[row[i]].append(i)
        for v_ in list( cl.values() ) :
            if tuple(v_) not in CL:
                CL[tuple(v_)] = d
    if bStrKeys :
        L = {}
        for item in CL.items():
            L['.'.join( lint2lstr(item[0])  )] = item[1]
        CL = L
    return ( {'CL':CL , 'F':F} )


def cluster_connections (       distm:np.array  , d:np.array ,
                                command = 'min' , Z:np.array=None ,
                                bReturnZ = False ) -> np.array :
    #
    from scipy.spatial.distance  import squareform
    from scipy.cluster.hierarchy import linkage as sclinks
    from scipy.cluster.hierarchy import fcluster
    if Z is None :
        Z       = sclinks( squareform(distm) , {'min':'single','max':'complete'}[command] )
    row         = fcluster ( Z ,d, 'distance' )
    if bReturnZ :
        return ( Z )
    return ( row )


def nearest_neighbor_graph_matrix ( distm:np.array, nn:int , bCheckSolution=False )->np.array:
    desc__ = """
     SLOW METHOD FOR CONSTRUCTING A NEAREST NEIGHBOR GRAPH
     REPRESENTATION OF A DISTANCE MATRIX GIVEN THAT EACH
     NODE SHOULD ONLY BE CONNECTED TO N NEIHGBOURS
     A ROW CORRESPONDS TO A SPECIFIC NODES NEIGHBOURS
     NOTE : THIS CREATES A NON-SYMMETRIC DISTANCE MATRIX
    """
    from scipy.spatial.distance import squareform
    if len(np.shape(distm)) == 1 :
        distm = squareform ( distm )
    nn_distm = []
    global_cval = -np.inf
    for row in distm :
        cval = sorted(row)[nn-1]
        if cval > global_cval :
            global_cval = cval
        nrow = np.array([ rval if rval<=cval else np.inf for rval in row ])
        nn_distm.append( nrow )
    if bCheckSolution :
        print (desc__ , '\n' , np.sum(np.array(nn_distm)<np.inf,1)  )
    return ( np.array(nn_distm), global_cval )


def CCA_DBSCAN ( eps:float, A:np.array , min_samples=1 ) -> np.array :
    # SKLEARN DBSCAN WORKS ON COORDINATES NOT THE DISTANCE MATRIX AS INPUT
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN( eps=eps , min_samples=min_samples ).fit(A)
    return ( clustering.labels_ )

def scipylinkages ( distm , command='min' , bStrKeys=True ) -> dict :
    from scipy.spatial.distance  import squareform
    from scipy.cluster.hierarchy import linkage as sclinks
    from scipy.cluster.hierarchy import fcluster
    Z = sclinks( squareform(distm) , {'min':'single','max':'complete'}[command] )
    CL = {}
    for d in Z[:,2] :
        row = fcluster ( Z ,d, 'distance' )
        sv_ = sorted(list(set(row)))
        cl  = {s:[] for s in sv_}
        for i in range( len( row ) ) :
            cl[row[i]].append(i)
        for v_ in list( cl.values() ) :
            if tuple(v_) not in CL:
                CL[tuple(v_)] = d
    if bStrKeys :
        L = {}
        for item in CL.items():
            L['.'.join( lint2lstr(item[0])  )] = item[1]
        CL = L
    return ( CL )

def linkages ( distm:np.array , command:str='min' ,
               bStrKeys:bool = True , bUseScipy:bool = False ,
               bMemSec=True, bLegacy:bool=False ) -> dict :
    distm = np.array(distm)
    if bMemSec :
        distm = distm.copy()
    if bUseScipy :
        linkages_ = scipylinkages ( distm ,command=command , bStrKeys = False )
    else :
        linkages_ = linkages_tiers ( D = distm , method = command )
    if bLegacy :
        return ( linkage( distm = distm , command = command ) )
    if bStrKeys :
        L = {}
        for item in linkages_.items():
            L['.'.join( lint2lstr(item[0])  )] = item[1]
        linkages_ = L
    return ( linkages_ )

def create_cluster_lookups ( cluster_ids:list[int] ) -> dict :
    pid2cid = {}
    cid2pid = {}
    for pid,cid in zip ( range(len(cluster_ids)) , cluster_ids ) :
        pid2cid[pid] = cid
        if cid in cid2pid :
            cid2pid[cid] .append(pid)
        else :
            cid2pid[cid] = [pid]
    return ( {'c2p':cid2pid ,'p2c':pid2cid } )

def pair_compactness (  distm:np.array , bSelected:list[bool] = None,
                        cluster1_indices:list[int] = None ,
                        cluster2_indices:list[int] = None  ) -> list[float] :
    if not bSelected is None :
        bNot = lambda x: [ not x_ for x_ in x ]
        g1   = np.where(bSelected)[0]
        g2   = np.where(bNot(bSelected))[0]
    else :
        g1   = cluster1_indices
        g2   = cluster2_indices
    #
    # BELOW SELECTION MUST BE FASTER
    allPairs = lambda A, v1, v2 : np.array([ A[i,j] for i in v1 for j in v2 ])
    #
    if len(g2) == 0 :
        g2 = g1
    if len(g1) == 0 :
        g1 = g2
    #
    pS		= np.sum( allPairs(distm,g1,g1) )
    qS		= np.sum( allPairs(distm,g2,g2) )
    pqS		= np.sum( allPairs(distm,g1,g2) )
    FS          = np.sum( allPairs(distm,[*g1,*g2],[*g1,*g2]) )
    p1122	= 1/len(g1)**2 * pS + 1/len(g2)**2 * qS
    p1212	= 2*1/len(g1) * 1/len(g2) * pqS # p2121 = p1212
    score	= p1212 - p1122
    norm_scores = np.array([pS,qS,pqS])/FS
    norm_scores = np.append(norm_scores, norm_scores[0] + norm_scores[1] - norm_scores[2] )
    #
    #	"QUADRATURE"	"TORQUE"	"NORMED DIFF"
    return ( [ p1212 - p1122 , pqS/len(g1)/len(g2) - (qS/len(g1)/2 + pS/len(g2)/2 )*0.5 , norm_scores ] )


def complete_compactness_score ( distm:np.array , cluster_ids:np.array ) :
    lucids = list(set(cluster_ids))
    lookup = { j:i for i,j in zip( range(len(lucids)),lucids ) }	# DID THE USER USE STRANGE INTS?
    res_l  = create_cluster_lookups( cluster_ids )			# YES WE NEED BOTH
    Z      = np.zeros( len(lucids)*len(lucids) ).reshape( len(lucids),len(lucids) )
    Y	   = np.zeros( len(lucids)*len(lucids) ).reshape( len(lucids),len(lucids) )
    X	   = np.zeros( len(lucids)*len(lucids) ).reshape( len(lucids),len(lucids) )
    density_functional = lambda x:np.mean(x)
    #
    for cid1 in lucids :
        for cid2 in lucids :
            xid1 = lookup[cid1]
            xid2 = lookup[cid2]
            if xid1 < xid2:
                cluster1_ids = res_l['c2p'][cid1]
                cluster2_ids = res_l['c2p'][cid2]
                res = pair_compactness ( distm , cluster1_indices=cluster1_ids , # BOTTLENECK
                                                 cluster2_indices=cluster2_ids ) # THIS CALCULATION IS SYMMETRIC
                score = res[0]
                erocs = res[1]
                nosco = res[2][-1]
            elif xid1 > xid2 :
                score = Z[xid2,xid1]
                erocs = Y[xid2,xid1]
                nosco = X[xid2,xid1]
            else :
                score = 0
                erocs = 0
                nosco = 0
            Z [ xid1 , xid2 ] = score
            Y [ xid1 , xid2 ] = erocs
            X [ xid1 , xid2 ] = nosco
    return ( [ Z , Y , X ] )

def split_or_merge ( distm:np.array, cluster_ids:np.array ) -> dict :
    density_functional = lambda x:np.mean(x)
    res = complete_compactness_score ( distm , cluster_ids ) # BOTTLENECK
    Z   = res[0]
    density_cutoff = density_functional ( [ Z[i,j] for i in range(len(Z)) for j in range(len(Z)) if not i==j ] )
    merge = Z[0]   <  density_cutoff
    split = Z[0]   >= density_cutoff
    keep  = res[2][-1] <= 0
    desc_ ="""
    compactness, torque, normed scores	- all information retained
    keep			- strong suggestion
    """
    return ( { 'compactness':Z , 'torque':res[1] ,'normed scores':res[2][-1] , 'keep split':keep , 'desc':desc_ } )


def cluster_appraisal( x:pd.Series , garbage_n = 0 , Sfunc = lambda x:np.mean(x,0)) -> list :
    """
Clustering Optimisation Method for Highly Connected Data
Richard Tjörnhammar
https://arxiv.org/abs/2208.04720v2
    """
    from collections import Counter
    decomposition = [ tuple((item[1],item[0])) for item in Counter(x.values.tolist()).items() if item[1] > garbage_n ]
    N     = len ( x )
    A , B = 1 , N
    if len(decomposition) > 0 :
        A = Sfunc(decomposition)[0]
        B = len(decomposition)
    else :
        decomposition = [ tuple( (0,0) ) ]
    decomposition .append( tuple( ( B ,-garbage_n )) )
    decomposition .append( tuple( ( A , B )) )
    decomposition .append( tuple( ( A*B/(A+B) , None ) ) )
    return ( decomposition )

def generate_clustering_labels ( distm:np.array , cmd:str='min' , labels:list[str]=None ,
                                 bExtreme:bool=False , n_clusters:int = None, Sfunc = lambda x:np.mean(x,0) ) -> tuple :
    clabels_n , clabels_o = None , None
    res         = sclinkages( distm , cmd )['F']
    index       = list( res.keys() )
    if labels is None :
        labels = range(len(distm))
    hierarch_df = pd.DataFrame ( res.values() , index=list(res.keys()) , columns = labels )
    cluster_df  = hierarch_df .T .apply( lambda x: cluster_appraisal(x , garbage_n = 0 , Sfunc=Sfunc ) )
    clabels_o , clabels_n = None , None
    screening    = np.array( [ v[-1][0] for v in cluster_df.values ] )
    Avals        = np.array( [ v[-2][0] for v in cluster_df.values ] )
    Bvals        = np.array( [ v[-2][1] for v in cluster_df.values ] )
    level_values = np.array( list(res.keys()) )
    if bExtreme :
        imax            = np.argmax( screening )
        clabels_o       = hierarch_df.iloc[imax,:].values.tolist()
    if not n_clusters is None :
        jhit = np.argmin([ np.abs(len(cluster_df.iloc[i])-2-n_clusters)\
                   for i in range(len(cluster_df)) ])
        clabels_n = hierarch_df.iloc[jhit,:].values.tolist()
    return ( clabels_n , clabels_o , hierarch_df , np.array( [ level_values , screening , Avals , Bvals ] ) )

def sL( L:list[str] ) -> pd.DataFrame :
    n = len(L)
    m = n
    K = np.meshgrid( range(n),range(m) ) [ 0 ] # WASTEFUL
    return ( pd.DataFrame( np.array([ L[i] for i in K.reshape(-1) ]).reshape(n,m) ) +'.'+\
             pd.DataFrame( np.array([ L[i] for i in K.reshape(-1) ]).reshape(n,m) ).T )

def sB( L:list[str], logic=lambda a,b : a==b ) -> list[bool] :
    convert_labels = lambda T : [ {s:i for s,i in zip(  list(set(T)),range(len(set(T)))  ) }[l] for l in T ]
    L = convert_labels(L)
    n = len(L)
    m = n
    K = np.meshgrid( range(n),range(m) ) [ 0 ] # WASTEFUL
    return (  logic( np.array([ L[i] for i in K.reshape(-1) ]).reshape(n,m) ,
             np.array([ L[i] for i in K.reshape(-1) ]).reshape(n,m).T ) )

def pdB( L:list[str] , logic = lambda a,b:a==b , type_function = lambda x:int(x) ) -> list[int] :
    return ( pd.DataFrame( sB(L=L,logic=logic) ).apply(lambda x:x.apply(type_function) ) )

def sB12( L1:list[str],L2:list[str] ) -> list[bool] :
    n = len(L1)
    m = len(L2)
    K = np.meshgrid( range(n),range(m) )
    return ( np.array([ L1[i] for i in K[0].reshape(-1) ]).reshape(n,m) ==\
             np.array([ L2[i] for i in K[1].reshape(-1) ]).reshape(n,m).T )

def label_correspondances ( L1:list[str] , L2:list[str] ,
                            bSymmetric:bool = False , bVerbose:bool = False,
                            bSingleton:bool = False ) -> list[int] :
    if bVerbose:
        print ( "EVALUATES GROUPING INTERACTIONS SO SINGLETON LABELING CAN INCLUDED" )
        print ( "INCLUDE IT BY SETTING bSingleton=True OR False TO EXCLUDE THE DIAGONAL" )
    nssum = lambda x:np.sum(np.sum(x))
    if len(L1) != len(L2) :
        print ( "WARNING : UNEQUAL LENGTHS DOESN'T MAKE SENSE" )
        return ( [0,0,0,0] )
    E = 1 - np.eye(len(L1))
    if bSingleton :
        E = 1  
    p_eS1 = pdB( L1 , logic = lambda a,b: a==b ) * E
    p_eS2 = pdB( L2 , logic = lambda a,b: a==b ) * E
    TP = nssum( ( p_eS1 ) * ( p_eS2 ) > 0)      # TP
    FN = nssum( ( 1-p_eS1 ) * (p_eS2) > 0)      # FN
    FP = nssum( ( p_eS1 ) * (1-p_eS2) > 0)      # FP
    TN = nssum( ( 1-p_eS1 ) * (1-p_eS2) > 0)    # TN
    return ( [TP,FP,FN,TN] )
#
# BEGIN DUNN
def delta(ck:np.array, cl:np.array)->np.array:
    values = np.ones([len(ck), len(cl)])*1E4
    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i]-cl[j])
    return np.min(values)
#
def big_delta(ci:np.array)->np.array:
    values = np.zeros([len(ci), len(ci)])
    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i]-ci[j])
    return np.max(values)
#
def dunn_index ( k_coordinates:list[np.array] ) -> float :
    deltas = np.ones([len(k_coordinates), len(k_coordinates)])*1E6
    big_deltas = np.zeros([len(k_coordinates), 1])
    l_range = list(range(0, len(k_coordinates)))
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta(k_coordinates[k], k_coordinates[l])
        big_deltas[k] = big_delta(k_coordinates[k])
    di = np.min(deltas)/np.max(big_deltas)
    return ( di )

import scipy.stats as sts
import scipy.spatial.distance as scd
import scipy.special as scp
#
def another_metric ( labels:list , D:np.array ) -> float :
    print ( 'WARNING : DONT USE THIS' )
    N	= len( labels )
    ldf	= pd.DataFrame( labels , columns=['L'] , index=range(N) )
    paired	= ldf.groupby('L').apply(lambda x:x.index.values.tolist())
    nm = np.shape(D)
    Z  = np.diag(-1*np.ones(N))
    if nm[0] != nm[1] or len(nm) == 1 :
        from scipy.spatial.distance import pdist,squareform
        D = squareform(D)
        if N != len(D) :
            print ('MALFORMED INPUT')
            exit(1)
    for v in paired.values :
        for i in range(len(v)):
            for j in range(i+1,len(v)):
                Z[ v[i] , v[j] ] =  1
                Z[ v[j] , v[i] ] = -1
    g1 , g2 = [] , []
    for z , d in zip(Z.reshape(-1),D.reshape(-1)) :
        if z  > 0 :
            g1.append(d) # IN
        elif z > -1 and z < 1 :
            g2.append(d) # OUT
    K , M = len(g1) , len(g2)
    results = sts.mannwhitneyu(  g1 , g2  )
    return ( 1-np.sqrt(results[0])/K/M , results[1] )

def hm_auc ( X:np.array = np.array([[33,6,6,11,2],[3,2,2,11,33]]) ,
             bReturnRates:bool=False ) :
    desc_ = """
	CALCULATES THE APPROXIMATE AUC AND STANDARD ERROR OF A SEGMENTATION
	STRATEGY WHERE IN AND OUT VALUES CORRESPOND TO SEVERITY DEGREE
	KNOWN IN LITT.
    """
    nN , nA	= np.sum(X,1)
    nT		= np.sum(X,0)
    N		= np.sum(X)
    E1	= X[0]
    E4	= np.array( [ *[0],*np.cumsum(X[0])[ :-1] ] )
    E3	= X[1]
    E2	= np.array( [ *np.cumsum(X[1][::-1])[::-1][1:],*[0]] )
    E5	= E1*E2 + 0.5 * ( E1*E3 )
    tE5 = np.sum(E5)
    W   = tE5/nA/nN
    AUC = W # THETA = W
    E6  = E3 * ( E4**2 + E4 * E1 + 1/3 *E1**2 )
    E7  = E1 * ( E2**2 + E2 * E3 + 1/3 *E3**2 )
    tE6 = np.sum(E6)
    tE7 = np.sum(E7)
    Q2	= tE6 / ( nA * nN**2 )
    Q1	= tE7 / ( nN * nA**2 )
    SETHETA = np.sqrt( ( W*(1-W) + (nA-1)*(Q1-W**2) + (nN-1)*(Q2-W**2) )/nA/nN )
    if bReturnRates :
        # W IS AN ANALYTIC AUC MEASURE
        # SETHETA IS THE ASSOCIATED STANDARD ERROR
        # num_auc IS A NUMERICAL ESTIMATE
        tpr = np.cumsum( X[0]/np.sum(X[0]) )
        fpr = np.cumsum( X[1]/np.sum(X[1]) )
        num_auc = np.trapz(tpr,fpr)
        num_err = np.abs(W-num_auc)
        return ( W , SETHETA , tpr , fpr , num_auc , num_err) 
    return ( W , SETHETA )

def num_auc ( X:np.array = np.array( [[33,6,6,11,2] , [3,2,2,11,33]]) ) -> tuple :
    # W IS AN ANALYTIC AUC MEASURE
    # SETHETA IS THE ASSOCIATED STANDARD ERROR
    # num_auc IS A NUMERICAL ESTIMATE
    tpr = np.cumsum( X[0]/np.sum(X[0]) )
    fpr = np.cumsum( X[1]/np.sum(X[1]) )
    num_auc = np.trapz(tpr,fpr)
    return ( tpr , fpr , num_auc )

def reformD( D:np.array , N:int ) -> np.array :
    from scipy.spatial.distance import pdist,squareform
    D = squareform(D)
    if N != len( D ) :
        print ('MALFORMED INPUT')
        exit( 1 )
    return ( D )

def approximate_auc(  labels:list , D:np.array , fraction:float = 0.1 , bVerbose=False , bInvert:bool=False , bLegacy:bool=False ) -> tuple :
    from scipy.stats			import rankdata
    from scipy.special			import binom
    from scipy.spatial.distance		import squareform
    #
    desc_ = """
 WHAT YOU HAVE IS A HISTOGRAM OF VALUES ACROSS A COMMON RANGE FOR THE IN AND OUT SET
 SO FOR CLUSTERING IT WOULD CORRESPOND TO HAVING THE VALUES BINNED ACROSS A COMMON SCALE
 IN THE DISTANCE MATRIX. DISTANCES ARE SMALLER IF SIMILAR NOT LARGER SO IN IS OUT IF
 YOU ARE USING SIMILIARTY VALUES THAT ARE INCREASING (I.E. COVARIANCE) THEN YOU MUST INVERT

 ALL TO ALL AUC CONTRIBUTION CALCULATION
    """
    M		= len(labels)
    nm          = np.shape ( D )
    if len( nm ) == 1 :
        D = reformD( D,M )
    if nm[0] != nm[1] :
        D = reformD( D,M )
    nm = np.shape(D)
    if M != nm[0] :
        print ( 'MALFORMED INPUT' )
        exit(1)
    #
    N	= nm[0]
    L	= int( np.round(N*fraction) ) + 1
    lD	= squareform( D )
    if bVerbose : # CHECKING
        pid2lid = lambda n,i,j : np.nan if i==j else int(binom(n,2) - binom(n-i,2) + (j-i-1) ) if i<j else int(binom(n,2) - binom(n-j,2) + (i-j-1) )
        i,j = 1,2
        print ( D[i,j] , lD[ pid2lid(N,i,j) ] )
        i,j= 2,1
        print ( D[i,j] , lD[ pid2lid(N,i,j) ] )
        print (  hanley_mcneil_auc( ) )
    ldf = pd.DataFrame( labels , columns=['L'] , index=range(N) )
    paired      = ldf.groupby('L').apply(lambda x:x.index.values.tolist())
    Z  = np.diag(-1*np.ones(N)) * 0.0
    for v in paired.values :
        for i in range(len(v)):
            for j in range(i+1,len(v)):
                Z[ v[i] , v[j] ] = 1
                Z[ v[j] , v[i] ] = 1
    lZ = squareform(Z)
    if bVerbose :
        print ( lZ )
    rlD = rankdata(lD,'average')
    mra = np.max( rlD )
    K	= int(np.floor(mra/L))+1
    if bVerbose :
        print ( rlD )
        print ( mra , L , K , int(np.floor(mra/L)) )
    X = np.zeros(2*K).reshape(2,K)
    for v,b in zip(rlD,lZ) :
        X[ 1-int(b) if not bInvert else int(b) ][ int(np.floor(v/L)) ] += 1
    if bVerbose :
        print ( X )
    if bLegacy :
        auc = hm_auc( X )
        if bVerbose :
            print ( auc )
        return ( auc )
    tpr_fpr = np.cumsum( X.T/np.sum(X,1) , 0 )
    TP = np.cumsum( X[0] )		# AT THRESHOLD
    FN = np.cumsum( X[0][::-1] )[::-1]	# AT THRESHOLD
    FP = np.cumsum( X[1] )		# AT THRESHOLD
    TN = np.cumsum( X[1][::-1] )[::-1]	# AT THRESHOLD
    auc = np.trapz(tpr_fpr[:,0],tpr_fpr[:,1])
    if str( auc ) == str( np.nan ) :
        auc = 0.0
    return ( auc , tpr_fpr, TP , TN , FN , FP )
#
def immersiveness ( I:int, labels:list , D:np.array , fraction:float = 0.1 , bVerbose=False , bInvert:bool=False ) -> tuple :
    desc_ = """
 WHAT YOU HAVE IS A HISTOGRAM OF VALUES ACROSS A COMMON RANGE FOR THE IN AND OUT SET
 SO FOR CLUSTERING IT WOULD CORRESPOND TO HAVING THE VALUES BINNED ACROSS A COMMON SCALE
 IN THE DISTANCE MATRIX. DISTANCES ARE SMALLER IF SIMILAR NOT LARGER SO IN IS OUT IF
 YOU ARE USING SIMILIARTY VALUES THAT ARE INCREASING (I.E. COVARIANCE) THEN YOU MUST INVERT

 ONE TO ALL AUC CONTRIBUTION CALCULATION
    """
    from scipy.stats                    import rankdata
    from scipy.special                  import binom
    from scipy.spatial.distance         import squareform
    M           = len(labels)
    nm          = np.shape ( D )
    if len( nm ) == 1 :
        D = reformD( D,M )
    if nm[0] != nm[1] :
        D = reformD( D,M )
    N	= M
    L	= int( np.round(N*fraction) ) + 1
    d		= rankdata( D[I] ,'average' )
    mrd		= np.max( d )
    bSel	= np.array( [ np.array([ l == labels[I] ,  l != labels[I] ]) for l in labels ] ).T
    if bVerbose :
        print ( d[bSel[0]] )
        print ( d[bSel[1]] )
    K	= int(np.floor(mrd/L))+1
    X	= np.zeros(2*K).reshape(2,K)
    for v,b in zip( d,bSel[0] ) :
        X[ 1-int(b) if not bInvert else int(b) ][ int(np.floor(v/L)) ] +=1
    auc = hm_auc( X )
    if bVerbose :
        print ( X )
        print ( auc )
    return ( auc )

def complete_immersiveness ( labels:list , D:np.array , fraction:float = 0.1 ,  bins:int = None ,
        bVerbose=False , bInvert:bool=False , bAlternative:bool = False ) -> np.array :
    if bAlternative :
        from impetuous.quantification import dirac_matrix
        bSanityCheck    = bVerbose
        if bins is None :
            bins = len(D)
            if bins<10 :
                bins = 100
        bins += 10
        grouped		= dirac_matrix( labels )
        ungrouped	= 1 - grouped
        if bSanityCheck :
            in_d		= [ d for d in ( D * grouped ) .reshape(-1) if d>0 ]
            out_d		= [ d for d in (D * ungrouped) .reshape(-1) if d>0 ]
            all_positives	= np.histogram(in_d , range=(0,np.max(D.reshape(-1))) , bins=bins )[0]
            all_positives	= all_positives/np.sum(all_positives)
            all_negatives , common_range = np.histogram(out_d, range=(0,np.max(D.reshape(-1))) , bins=bins )
            all_negatives 	= all_negatives/np.sum(all_negatives)
            domain = 0.5*(common_range[1:]+common_range[:-1])
            #
            tpr = np.cumsum(all_positives)
            tpr = tpr/tpr[-1]
            fpr = np.cumsum(all_negatives)
            fpr = fpr/fpr[-1]
            print ( 'FULL AUC:' , np.trapz(tpr,fpr) )
        in_d            = D * grouped
        out_d           = D * ungrouped
        dr		= np.max(D.reshape(-1))/bins
        domain		= np.array( [ dr*0.5+i*dr for i in range( bins ) ] )
        cP,cN = [],[]
        for d in domain :
            cP.append ( np.sum( (in_d>0 ) * (in_d<=d) , 0 ) )
            cN.append ( np.sum( (out_d>0) * (out_d<=d) , 0 ) )
        cPdf = pd.DataFrame(cP) + 1
        cNdf = pd.DataFrame(cN) + 1
        TPR = (cPdf/cPdf.iloc[-1,:]).values
        FPR = (cNdf/cNdf.iloc[-1,:]).values
        del cPdf,cNdf
        all_immersions = []
        for tpr,fpr in zip( TPR.T,FPR.T ) :
            all_immersions.append( ( np.trapz( tpr,fpr ), np.abs(np.trapz(np.diff(tpr),np.diff(fpr)) ) ) )
        if bSanityCheck:
            import matplotlib.pyplot as plt
            plt.subplot(211)
            plt.plot( domain[1:] , np.diff(TPR[:,0]) 	, 'y' , label = 'Grouped, first instance, immersion'   	)
            plt.plot( domain[1:] , np.diff(FPR[:,0]) 	, 'c' , label = 'Ungrouped, first instance, immersion' 	)
            plt.plot( domain 	 , all_positives 	, 'r' , label = 'Grouped, all instances, Immersiveness'	)
            plt.plot( domain 	 , all_negatives 	, 'b' , label = 'Ungrouped, all instances, Immersiveness' )
            plt.title( 'Immersiveness and Immersion' )
            plt.xlabel( 'Distance [Arb Unit]')
            plt.ylabel( 'Histogram' )
            plt.legend()
            plt.subplot(212)
            mFPR = np.mean( FPR.T , 0 )
            mTPR = np.mean( TPR.T , 0 )
            sFPR = np.std(FPR.T,0)
            sTPR = np.std(TPR.T,0)
            #
            plt.plot( mFPR - sFPR , mTPR+sTPR , ':g'	, label = 'mean + error'  )
            plt.plot( mFPR , mTPR , 'g' , label='mean' )
            plt.plot( mFPR + sFPR , mTPR - sTPR  , ':g'	, label = 'mean - error' )
            #
            plt.xlabel( 'FPR' )
            plt.ylabel( 'TPR' )
            plt.legend()
            str_conv= lambda num: str(np.round(num*1000)/1000)
            cal_title = 'Full AUC and error estimate: ' +    str_conv(   np.trapz( mTPR , mFPR )) +\
			' +- ' + str_conv ( np.trapz( mTPR + sTPR ,  mFPR - sFPR ) -\
                        np.trapz( mTPR - sTPR ,  mFPR + sFPR ) )
            plt.title( cal_title )
            plt.show()
            print ( cal_title )
        return ( np.array( all_immersions ) )
    total_immersiveness	= []
    N 			= len(labels)
    for i in range ( N ) :
        immersed = immersiveness ( i , labels , D )
        if bVerbose :
            print ( 'IMERSIVENESS = ' , i ,   immersed   )
        total_immersiveness .append( np.array(immersed) )
    return ( np.array ( total_immersiveness ) )
#
#
def happiness ( entry_nr:int , ldf:pd.DataFrame , ndf:pd.DataFrame , clustercol:int=0 , namecol=0 , neighborcol=1 , lookup:dict=None ) -> float :
    descr_ = """
 ldf ASSUMES THE CLUSTER INFORMATION HAS UNIQUE NAMES AS INDEX AND CLUSTER IDS IN clustercol
 ndf ASSUMES THAT namecol CONTAINS UNIQUE NAMES AS VALUES FOUND IN THE ldf INDEX AND THAT NEIGHBOR
	COL CONTAINS UNIQUE NAMES. ndf CONTAINS SELF AS ITS FIRST NEIGHBOR.
    """
    if lookup is None :
        lookup  = { i:(c,j) for i,c,j in zip(ldf.index.values,ldf.iloc[:,0].values,range(len(ldf)) ) }
    i_ = entry_nr
    I , C = ldf.index.values[i_],ldf.iloc[i_,clustercol]
    check       = [ [lookup[name],name==I] for name in ndf.iloc[ndf.iloc[:,0].values==I].iloc[:,1].values  ]
    happy       = [ c[0][0] for c in check if c[1] ][0]
    happiness   = np.sum( [ c[0][0]==happy for c in check ] )/len(check)
    return ( happiness )

def complete_happiness ( ldf:pd.DataFrame , ndf:pd.DataFrame=None , clustercol:int=0 , namecol=0 , neighborcol=1  ) -> list[float] :
    if ndf is None :
        from impetuous.quantification import dirac_matrix
        N = float( len(set(ldf.iloc[:,clustercol].values.tolist())) )
        N = len(ldf.index)
        return( (np.sum( dirac_matrix(ldf.iloc[:,clustercol].values ), 0 )/N).tolist() )
    lookup = { i:(c,j) for i,c,j in zip(ldf.index.values,ldf.iloc[:,0].values,range(len(ldf)) ) }
    total_happiness_ = []
    for i_ in range( len(ldf) ) :
        total_happiness_ .append( happiness( i_ , ldf , ndf , lookup=lookup ) )
    return ( total_happiness_ )
#

if __name__ == '__main__' :

    D = [[0,9,3,6,11],[9,0,7,5,10],[3,7,0,9,2],[6,5,9,0,8],[11,10,2,8,0] ]
    print ( np.array(D) )
    print ( linkage( D, command='min') )
    print ( linkage( D, command='max') )

    if False :
        #
        # TEST DEPENDS ON THE DIABETES DATA FROM BROAD INSTITUTE
        filename = './Diabetes_collapsed_symbols.gct'
        df_ = pd.read_csv(filename,'\t',index_col=0,header=2)
        ddf = df_.loc[:,[ col for col in df_.columns if '_' in col ]]
        ddf .index = [idx.split('/')[0] for idx in ddf.index]
        run_clustering_and_write_gmt( ddf , clustering_algorithm )
        #
        CLU = Cluster( )
        CLU .approximate_density_clustering(ddf)
        CLU .write_gmt()

    if True :
        A = np.array( [ [0.00, 0.10, 0.10, 9.00, 9.00, 9.00],
                        [0.10, 0.00, 0.15, 9.00, 9.00, 9.00],
                        [0.10, 0.15, 0.00, 9.00, 9.00, 9.00],
                        [9.00, 9.00, 9.00, 0.00, 0.10, 0.10],
                        [9.10, 9.00, 9.00, 0.10, 0.00, 0.15],
                        [9.10, 9.00, 9.00, 0.10, 0.15, 0.00] ] )
        print ( connectivity(A,0.11) )
        print ( dbscan(distance_matrix=pd.DataFrame(A).values,eps=0.11,minPts=2) )

    import time
    print ( 'HERE' )

    coord_fn = 'NAJ-buckle.xyz'
    coords = read_xyz ( fname=coord_fn,sep=' ' )

    crds = [ c[1] for c in coords ]
    distm = np.array([ np.sqrt(np.sum((np.array(p)-np.array(q))**2)) for p in crds for q in crds ] ).reshape(len(crds),len(crds))
    Self = NodeGraph()

    for a in range( 100 , 165 ) :
        T = []
        alpha = a/100.
        T.append( time.time() )
        R = Self.connectivity( distm , alpha )
        T.append( time.time() )
        P = connectivity  ( distm , alpha )
        T.append( time.time() )
        Q = connectedness ( distm , alpha )
        T.append( time.time() )
        print ( len(R) , len( P[0] ) , len(Q) , np.diff(T) , alpha )


    coordinates = np.array([ *np.random.randn(500,100) , *(np.random.randn(500,100) + 10) ])
    print ( coordinates )
    dm = distance_calculation( coordinates ,
        distance_type = 'correlation,spearman' ,
        #bRemoveCurse  = True , nRound=4 )
        bRemoveCurse = False )
    print ( dm )
    print ( np.sum(np.diag(dm)) )
    cln,clo,hdf,sol = generate_clustering_labels( dm , cmd='max' , bExtreme=True , n_clusters=80 )
    imax = np.argmax( sol[1] )
    print ( clo )
    print ( sol[0] )
    print ( sol[1] )
    import matplotlib.pyplot as plt
    plt.plot(sol[0],sol[1],'k')
    plt.show()


    """    CONNECT +
 N  M  K   EDNESS LINEAR    IVITY JIT     EDNESS-JIT
67 67 67 [7.61270523e-04 5.55515289e-05 2.30550766e-04] 1.01
63 63 63 [7.44342804e-04 4.60147858e-05 2.15530396e-04] 1.02
63 63 63 [7.4672699e-04 4.5299530e-05 2.1481514e-04] 1.03
63 63 63 [7.39812851e-04 4.55379486e-05 2.14338303e-04] 1.04
62 62 62 [7.46965408e-04 4.62532043e-05 2.13384628e-04] 1.05
62 62 62 [7.49349594e-04 4.50611115e-05 2.13384628e-04] 1.06
62 62 62 [7.44104385e-04 4.52995300e-05 2.13384628e-04] 1.07
62 62 62 [7.45058060e-04 4.52995300e-05 2.12907791e-04] 1.08
56 56 56 [7.20024109e-04 4.86373901e-05 1.94549561e-04] 1.09
44 44 44 [6.73055649e-04 5.17368317e-05 1.59502029e-04] 1.1
44 44 44 [6.67810440e-04 5.07831573e-05 1.58548355e-04] 1.11
44 44 44 [6.70433044e-04 5.05447388e-05 1.59740448e-04] 1.12
44 44 44 [6.73770905e-04 5.10215759e-05 1.58309937e-04] 1.13
44 44 44 [6.69479370e-04 5.05447388e-05 1.58309937e-04] 1.14
44 44 44 [6.61611557e-04 5.00679016e-05 1.58071518e-04] 1.15
44 44 44 [6.54459000e-04 5.03063202e-05 1.58309937e-04] 1.16
44 44 44 [6.59942627e-04 5.07831573e-05 1.58071518e-04] 1.17
44 44 44 [6.54697418e-04 5.03063202e-05 1.58071518e-04] 1.18
44 44 44 [6.56127930e-04 4.98294830e-05 1.58786774e-04] 1.19
44 44 44 [6.54697418e-04 5.00679016e-05 1.57833099e-04] 1.2
44 44 44 [6.57558441e-04 5.07831573e-05 1.57833099e-04] 1.21
43 43 43 [6.46114349e-04 5.07831573e-05 1.55925751e-04] 1.22
43 43 43 [6.56843185e-04 5.10215759e-05 1.53303146e-04] 1.23
43 43 43 [6.49929047e-04 5.10215759e-05 1.53064728e-04] 1.24
43 43 43 [6.58512115e-04 5.12599945e-05 1.54018402e-04] 1.25
43 43 43 [6.58512115e-04 5.07831573e-05 1.53064728e-04] 1.26
43 43 43 [6.51597977e-04 5.05447388e-05 1.52349472e-04] 1.27
43 43 43 [6.55651093e-04 5.07831573e-05 1.52826309e-04] 1.28
43 43 43 [6.47783279e-04 5.05447388e-05 1.56879425e-04] 1.29
43 43 43 [6.46829605e-04 5.05447388e-05 1.52349472e-04] 1.3
42 42 42 [6.49929047e-04 5.10215759e-05 1.51872635e-04] 1.31
40 40 40 [6.52074814e-04 5.10215759e-05 1.48057938e-04] 1.32
36 36 36 [6.57796860e-04 5.29289246e-05 1.45435333e-04] 1.33
36 36 36 [6.46829605e-04 5.22136688e-05 1.44720078e-04] 1.34
35 35 35 [6.37769699e-04 5.14984131e-05 1.37805939e-04] 1.35
34 34 34 [6.27517700e-04 5.17368317e-05 1.38044357e-04] 1.36
34 33 34 [6.24418259e-04 5.34057617e-05 1.39951706e-04] 1.37
33 32 33 [6.37531281e-04 5.38825989e-05 1.43051147e-04] 1.38
31 29 31 [6.30378723e-04 5.10215759e-05 1.36613846e-04] 1.39
30 28 30 [6.11305237e-04 5.10215759e-05 1.29461288e-04] 1.4
28 28 28 [6.01291656e-04 5.22136688e-05 1.32083893e-04] 1.41
28 28 28 [5.98907471e-04 5.24520874e-05 1.33991241e-04] 1.42
23 23 23 [5.91039658e-04 5.26905060e-05 1.19209290e-04] 1.43
20 20 20 [5.77926636e-04 5.24520874e-05 1.17063522e-04] 1.44
18 18 18 [5.57899475e-04 5.17368317e-05 1.04665756e-04] 1.45
18 18 18 [5.56707382e-04 5.14984131e-05 1.06573105e-04] 1.46
16 16 16 [5.44548035e-04 5.05447388e-05 9.01222229e-05] 1.47
15 15 15 [5.30004501e-04 4.98294830e-05 8.63075256e-05] 1.48
11 11 11 [5.12361526e-04 5.07831573e-05 6.98566437e-05] 1.49
11 11 11 [5.13076782e-04 5.05447388e-05 7.00950623e-05] 1.5
11 11 11 [5.07593155e-04 5.07831573e-05 6.98566437e-05] 1.51
11 11 11 [5.08546829e-04 5.05447388e-05 7.03334808e-05] 1.52
11 11 11 [5.11646271e-04 5.05447388e-05 7.12871552e-05] 1.53
9 9 9 [4.99248505e-04 4.93526459e-05 6.41345978e-05] 1.54
6 5 6 [4.94241714e-04 4.86373901e-05 5.86509705e-05] 1.55
5 5 5 [4.85181808e-04 4.88758087e-05 6.07967377e-05] 1.56
5 5 5 [4.85181808e-04 4.88758087e-05 6.03199005e-05] 1.57
5 5 5 [4.80413437e-04 4.88758087e-05 6.34193420e-05] 1.58
5 5 5 [4.87327576e-04 4.86373901e-05 6.00814819e-05] 1.59
5 5 5 [4.86612320e-04 4.86373901e-05 6.67572021e-05] 1.6
3 3 3 [4.74214554e-04 4.86373901e-05 5.53131104e-05] 1.61
1 1 1 [4.72545624e-04 4.83989716e-05 4.72068787e-05] 1.62
1 1 1 [4.73499298e-04 4.81605530e-05 4.67300415e-05] 1.63
1 1 1 [4.74929810e-04 5.17368317e-05 4.64916229e-05] 1.64
    """
