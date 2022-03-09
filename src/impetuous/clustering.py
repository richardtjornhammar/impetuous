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
        def distance_matrix_to_absolute_coordinates ( D , bSquared = False, n_dimensions=2 ):
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
                xr = np.dot( Z.T,Vt )
                return ( xr )
else :
        def distance_matrix_to_absolute_coordinates ( D , bSquared = False, n_dimensions=2 ):
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
                xr = np.dot( Z.T,Vt )
                return ( xr )

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

def linkage ( distm:np.array , command:str = 'max' ) -> dict :
    #
    # CALUCULATES WHEN SAIGAS ARE LINKED
    # MAX CORRESPONDS TO COMPLETE LINKAGE
    # MIN CORRESPONDS TO SINGLE LINKAGE
    #
    D = distm
    N = len(D)

    sidx = [ str(i)+'-'+str(j) for i in range(N) for j in range(N) if i<j ]
    pidx = [ [i,j] for i in range(N) for j in range(N) if i<j ]
    R = [ D[idx[0]][idx[1]] for idx in pidx ]

    label_order = lambda i,k: i+'-'+k if '.' in i else k+'-'+i if '.' in k else  i+'-'+k if int(i)<int(k) else k+'-'+i

    if command == 'max':
        func  = lambda R,i,j : [(R[i],i),(R[j],j)] if R[i]>R[j] else [(R[j],j),(R[i],i)]
    if command == 'min':
        func  = lambda R,i,j : [(R[i],i),(R[j],j)] if R[i]<R[j] else [(R[j],j),(R[i],i)]

    LINKS = {}
    cleared = set()

    for I in range( N**2 ) : # WORST CASE SCENARIO

        clear = []
        if len(R) == 0 :
            break

        nar   = np.argmin( R )
        clu_idx = sidx[nar].replace('-','.')

        LINKS = { **{ clu_idx : R[nar] } , **LINKS }
        lp    = {}

        for l in range(len(sidx)) :
            lidx = sidx[l]
            lp [ lidx ] = l

        pij   = sidx[nar].split('-')
        cidx  = set( unpack( [ s.split('-') for s in [ s for s in sidx if len(set(s.split('-'))-set(pij)) == 1 ] ] ) )-set(pij)
        ccidx = set( [ s for s in [ s for s in sidx if len(set(s.split('-'))-set(pij)) == 1 ] ] )
        found = {}

        i   = pij[0]
        j   = pij[1]

        for k in cidx :
            h0 , h1 , h2 , q0, J = None , None , None , None, 0
            if k == j or k == i :
                continue
            la1 = label_order(i,k)
            if la1 in lp :
                J+=1
                h1 = lp[ label_order(i,k) ]
                h0 = h1 #; q0 = i
            la2 = label_order(j,k)
            if la2 in lp :
                J+=1
                h2 = lp[ label_order(j,k) ]
                h0 = h2 #; q0 = j
            if J == 2 :
                Dijk = func ( R , h1 , h2 )
            elif J == 1 :
                Dijk = [[ R[h0],h0 ]]
            else :
                continue
            nidx = [ s for s in sidx[Dijk[0][1]].split('-') ]
            nidx = list( set(sidx[Dijk[0][1]].split('-'))-set(pij) )[0]
            nclu_idx = clu_idx + '-' + nidx
            found[ nclu_idx ] = Dijk[0][0]

        clear = [*[ lp[c] for c in ccidx ],*[nar]]
        cleared = cleared | set( unpack( [ sidx[c].split('-') for c in clear ]) )
        R = rem(R,clear)
        sidx = rem(sidx,clear)

        for label,d in found.items() :
            R.append(d)
            sidx.append(label)
    for c in sorted( [ (v,k) for k,v in LINKS.items() ] )[-1][1].split('.'):
        LINKS[c] = 0
    return ( LINKS )


if __name__=='__main__' :

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
        print ( connectivity(A,0.11) ) # 
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
