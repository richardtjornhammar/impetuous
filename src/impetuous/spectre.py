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
import pandas as pd
import numpy as np
from scipy.stats import rankdata

beta2M = lambda b : math.log2( (b/(1-b)) )
M2beta = lambda M : 2**M/(2**M+1)

map_df_to_spectral_order_domain = lambda df: \
	df .apply ( lambda x:(rankdata(x,'average')-0.5)/len(x) ) \
           .apply ( lambda B:[ beta2M(b) for b in B] )

power = lambda s : np.dot( s.T,np.conj(s) )
spectre_to_power = lambda sdf: \
	pd.DataFrame ( \
		power ( sdf.T.apply(np.fft.fft) ) , \
			index = sdf.index , columns = sdf.index \
		)

def transform_to_resonances( spectrum ) :
    # assumes spectrum is a square symmetric matrix
    f_ls = np.mean( np.abs(spectrum.iloc[0,:].quantile([0.01,0.99])) )
    reso = spectrum .apply(lambda ser: np.real(ser.to_numpy())) \
                .apply(lambda X:[x/f_ls for x in X]) .apply(M2beta) \
                .apply(lambda X:[ 2.*x-1. for x in X] )
    return ( reso )

if __name__ == '__main__' :
    print ( 'SORRY: NO TESTS HERE' )
    print ( 'DEVELOPMENTAL VERSION')
