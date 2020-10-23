"""
Copyright 2019 RICHARD TJÖRNHAMMAR

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
from impetuous.clustering import *
from impetuous.quantification import *
import sys

if __name__ == '__main__' :
    sys.path.append('../../')
    #
    # TEST DEPENDS ON THE DIABETES DATA FROM BROAD INSTITUTE
    df_ = pd.read_csv('./Diabetes_collapsed_symbols.gct','\t',index_col=0,header=2)
    ddf = df_.loc[:,[ col for col in df_.columns if '_' in col ]] ; ddf.index = [idx.split('/')[0] for idx in ddf.index]
    run_clustering_and_write_gmt(ddf,clustering_algorithm)

    CLU=Cluster()
    CLU.approximate_density_clustering(ddf)
    CLU.write_gmt('./data_driven_clusters.gmt')
