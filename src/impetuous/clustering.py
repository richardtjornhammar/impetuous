import sklearn.cluster as sc
import pandas as pd
import sys

clustering_algorithm = None
clustering_algorithm = sc.KMeans(10) # CHOOSE SOMETHING YOU LIKE NOT THIS

def run_clustering_and_write_gmt( df , ca , filename = './cluster_file.gmt' ) :
    labels = clustering_algorithm.fit_predict(df.values)
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

