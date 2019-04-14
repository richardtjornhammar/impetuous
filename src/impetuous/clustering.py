import sklearn.cluster as sc
import pandas as pd
import numpy as np
import sys

clustering_algorithm = None
clustering_algorithm = sc.KMeans(10) # CHOOSE SOMETHING YOU LIKE NOT THIS

class Cluster(object):
    def __init__( self, nbins=50 ) :
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from numpy import histogram2d
        from scipy.stats import rankdata
        self.nbins = nbins
        self.histogram2d = histogram2d
        self.KMeans = KMeans
        self.rankdata = rankdata
        self.pca_f = PCA(2)
        self.centroids_ = None
        self.labels_ = None
        self.df_ = None
        self.num_index_ = None

    def approximate_density_clustering( self, df, nbins=None ) :
        #
        # GENES APPROX 20K OK SO APPROX 50 BINS
        # ANALYTES ON ROWS, SAMPLE POINTS ON COLUMNS
        if nbins is None :
            nbins = self.nbins
        self.df_= df
        frac_df = df.apply( lambda x:self.rankdata( x , method='average' )/float(len(x)) )
        self.pca_f.fit(frac_df.T.values)
        vals,xe,ye = self.histogram2d(self.pca_f.components_[0],self.pca_f.components_[1],bins=nbins)
        mvs, svsx, svsy = np.mean(vals),np.std(vals,0),np.std(vals,1)
        svs = np.sqrt(svsx**2+svsy**2)
        #
        # IS THERE A DENSITY PEAK SEPARABLE FROM THE MEAN
        # SHOULD DO GRADIENT REJECTION BASED ON TTEST PVALUES
        hits = vals>mvs+0.5*svs

        xe_,ye_=0.5*(xe[:1]+xe[1:]),0.5*(ye[:1]+ye[1:])
        idx = np.where(hits); xi,yj = idx[0],idx[1]
        centroids = [ (xe[ri],ye[rj]) for (ri,rj) in zip(xi,yj) ]

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

def run_clustering_and_write_gmt( df , ca , filename = './approx_cluster_file.gmt' ) :
    labels = ca.fit_predict(df.values)
    llabs = [ l for l in labels ]; ulabs=set(llabs)
    with open(filename,'w') as of :
        for ulab in ulabs :
            analytes = df.iloc[llabs==ulab].index.values
            print ( 'CLU-'+str(ulab),'\tDESCRIPTION\t'+'\t'.join(analytes), file=of )

if __name__ == '__main__' :
    #
    # TEST DEPENDS ON THE DIABETES DATA FROM BROAD INSTITUTE
    df_ = pd.read_csv('./Diabetes_collapsed_symbols.gct','\t',index_col=0,header=2)
    ddf = df_.loc[:,[ col for col in df_.columns if '_' in col ]] ; ddf.index = [idx.split('/')[0] for idx in ddf.index]
    run_clustering_and_write_gmt(ddf,clustering_algorithm)

    CLU=Cluster()
    CLU.approximate_density_clustering(ddf)
    CLU.write_gmt()
