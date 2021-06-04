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

*) IN SHORT THE DISTANCE MATRIX IN THE CONVERSION ROUTINE BACK TO ABSOLUTE COORDINATES USES R2 DISTANCES.
"""

if bUseNumba :
	@jit(nopython=True)
	def distance_matrix_to_absolute_coordinates ( D , bSquared = False, n_dimensions=2 ):
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


#bUseNumba=False

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
		nr_sq,mr_sq = np.shape(B)
		if nr_sq != mr_sq :
			print ( 'ERROR:: FAILED' )
		N = mr_sq
		res, nvisi, s, NN, ndx, C = [0], [0], [0], [0], [0], 0
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


def connectivity_legacy001 ( B , val, bVerbose=False ) :
	description="""
This is a cutoff based clustering algorithm. The intended use is to supply a distance matrix and a cutoff value (then becomes symmetric positive).  For a small distance cutoff, you should see all the parts of the system and for a large distance cutoff, you should see the entire system. It has been employed for statistical analysis work as well as the original application where it was employed to segment molecular systems.
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
		exit (1)
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
        pd.DataFrame(dM1,index=rana_df.index,columns=rana_df.index).to_csv('../data/dM1.tsv','\t')
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
        pd.DataFrame(dM2,index=rana_df.columns,columns=rana_df.columns).to_csv('../data/dM2.tsv','\t')
        #
        # This algorithm is exact but scales somewhere between n^2 and n log n
        max_var_projection = distance_matrix_to_absolute_coordinates ( dM2 , n_dimensions=n_dimensions )
        cluster_coords_s = pd.DataFrame( max_var_projection ,
                                    columns = rana_df.columns ,
                                    index = dimcrdnames ).T
        cluster_coords_s.to_csv('../data/conclust_s.tsv','\t')

    return ( cluster_coords_f,cluster_coords_s )


if __name__ == '__main__' :

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

    if False :
        A = np.array( [ [0.00, 0.10, 0.10, 9.00, 9.00, 9.00],
        		[0.10, 0.00, 0.15, 9.00, 9.00, 9.00],
        		[0.10, 0.15, 0.00, 9.00, 9.00, 9.00],
        		[9.00, 9.00, 9.00, 0.00, 0.10, 0.10],
        		[9.10, 9.00, 9.00, 0.10, 0.00, 0.15],
        		[9.10, 9.00, 9.00, 0.10, 0.15, 0.00] ] )
        print( connectivity(A,0.01) )

