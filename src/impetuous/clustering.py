"""
Copyright 2020 RICHARD TJÖRNHAMMAR

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

import sklearn.cluster as sc
import pandas as pd
import numpy as np
import sys

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
    #
    data = []
    for (x,y,t,cl,co) in zip( x_pc1,y_pc2,synonyms , [cl for cl in CLUSTER.labels_] ,
                              [cluster_colors[cl] for cl in CLUSTER.labels_] ) :
        data.append([x,y,t,cl,co])
    clustering_df = pd.DataFrame( data , columns = ['X','Y','Type','Cluster','Color'])
    if not df is None :
        clustering_df.index =  df.index.values 
    clustering_df.to_csv( output_name , '\t' )
    return ( clustering_df )


def connectivity ( B , val, bVerbose=True ) :
	# PYTHON ADAPTATION OF MY C++ CODE THAT CAN BE FOUND IN
	# https://github.com/richardtjornhammar/RichTools/blob/master/src/cluster.cc
	# AROUND LINE 2277
	# CONSIDER COMPILING AND USING THAT AS A MODULE INSTEAD OF THIS SINCE IT IS
	# A LOT FASTER
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
				if ( B[i,j]<val ) :
					NN.append(j)
			while ( len(NN)>0 ) :
				# back pop_back
				k = NN[-1]; NN = NN[:-1]
				nvisi[k] = C
				for j in range(N):
					if ( B[j,k]<val ) :
						for q in range(N) :
							if ( nvisi[q] == j+1 ) :
								NN.append(q)
	if bVerbose : # VERBOSE
		print("INFO "+str(-1*C) +" clusters" )
	Nc = [ 0 for i in range(-1*C) ]
	for q in range(N) :
		res[  q*2+1 ] = q;
		res[  q*2   ] = nvisi[q]-C;
		Nc [res[q*2]]+= 1;
		if bVerbose :
			print ( " "+str(res[q*2])+" "+str(res[2*q+1]) )
	for i in range(-1*C) :
		print( "CLUSTER "  +str(i)+ " HAS " + str(Nc[i]) + " ELEMENTS")
	return ( Nc , np.array(res[:-1]).reshape(-1,2) )

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
        CLU = Cluster()
        CLU .approximate_density_clustering(ddf)
        CLU .write_gmt()

    if True :
        A = np.array( [ [0.00, 0.10, 0.10, 9.00, 9.00, 9.00],
        		[0.10, 0.00, 0.15, 9.00, 9.00, 9.00],
        		[0.10, 0.15, 0.00, 9.00, 9.00, 9.00],
        		[9.00, 9.00, 9.00, 0.00, 0.10, 0.10],
        		[9.10, 9.00, 9.00, 0.10, 0.00, 0.15],
        		[9.10, 9.00, 9.00, 0.10, 0.15, 0.00] ] )
        print( connectivity(A,1.) )


