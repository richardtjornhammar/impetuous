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
from scipy.stats import rankdata

beta2M = lambda b : math.log2( (b/(1-b)) )
M2beta = lambda M : 2**M/(2**M+1)
#
# FRACTIONAL RANKS ON [0,1] ARE RECAST TO +- INFINITY
map_df_to_spectral_order_domain = lambda df: \
        df .apply ( lambda x:(rankdata(x,'average')-0.5)/len(x) ) \
           .apply ( lambda B:[ beta2M(b) for b in B] )

power = lambda s : np.dot( s.T,np.conj(s) )
#
# THE SPECTRUM IS TRANSFORMED INTO A POWER SPECTRUM
spectre_to_power = lambda sdf: \
        pd.DataFrame ( \
                power ( sdf.T.apply(np.fft.fft) ) , \
                        index = sdf.index , columns = sdf.index \
                )
#
# THE BELOW METHOD IS HIGHLY EXPERIMENTAL
def transform_to_resonances( spectrum ) :
    print ( "WARNING WARNING" )
    # assumes spectrum is a square symmetric matrix
    f_ls = np.mean( np.abs(spectrum.iloc[0,:].quantile([0.01,0.99])) )
    reso = spectrum .apply(lambda ser: np.real(ser.to_numpy())) \
                .apply(lambda X:[x/f_ls for x in X]) .apply(M2beta) \
                .apply(lambda X:[ 2.*x-1. for x in X] )
    return ( reso )
#
# TIME AUTOCORRELATIONS ARE NOT THE SAME THING
# AS PEARSON AUTOCORRELATIONS
def calc_autocorrelation( tv , dt=1 ,bMeanCentered=True ) :
    # If you studied statistical mechanics you would
    # know about the Wiener Kinchin theorem
    if bMeanCentered :
        # So for stocks you might want
        # to remove the mean
        tv-=np.mean(tv)
    ftv = np.fft.fft(tv)
    rt  = np.fft.ifft(ftv*np.conj(ftv))
    rt  = rt/rt[0]
    rt  = rt[:int(np.floor(len(rt)*0.5))]
    return( [dt*t for t in range(len(rt))], [np.real(r) for r in rt] )

if __name__ == '__main__' :
    print ( 'SORRY: NO TESTS HERE' )
    print ( 'DEVELOPMENTAL VERSION')
