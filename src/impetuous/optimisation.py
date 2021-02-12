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
from scipy.stats import kurtosis

def get_coordinates ( values , length_scales=None ) :
    n , m = np.shape( values )
    if length_scales is None :
        L = [ n,m ]
    else :
        L = length_scales
    # ASSUMES 2D
    d1 = int ( (n-n%2)/2 )
    d2 = int ( (n+n%2)/2 )
    d3 = int ( (m-m%2)/2 )
    d4 = int ( (m+m%2)/2 )
    coordinates = np.mgrid[-d1:d2, -d3:d4]
    coordinates = np.array([ r/l for (r,l) in zip(coordinates,L)])
    return ( coordinates )

def convolve ( xi,R,bFlat = True ) :
    fval = np.fft.fft2( R[1])
    G    = np.exp( -( np.sum( np.array( R[0] )**2 ,0) )*( xi ) )    
    conn = np.fft.ifft2(np.fft.fftshift((np.fft.fftshift(fval)*G))).real
    if bFlat :
        conn = conn.reshape(-1)
    return ( conn )

def golden_ration_phasetransition_search ( values , coordinates = None ,
                           unimodal_function = lambda x:kurtosis(x) ,
                           convolution = lambda xi,R:convolve(xi,R) ,
                           length_scales = None , extreme = 1000.0, tol=1e-6 ):
    saiga__ = """
see my github.com/richardtjornhammar/MapTool repo i.e. file
maptool.cc  around line 845 or
maptool2.cc around line 646 
"""
    if coordinates is None:
        coordinates = get_coordinates(values,length_scales)
    R = [ coordinates,values ]
    
    golden_ratio    = ( 5.0**0.5-1.0 )*0.5
    a , b , fc , fd = -1.0*extreme, 1.0*extreme, 0.0, 0.0
    c , d =  b-golden_ratio*(b-a), a+golden_ratio*(b-a)
    metric, optimum = 0.0 , 0.0
    while ( d-c>tol ) :
        fc = unimodal_function( convolution ( c, R ) ) ;
        fd = unimodal_function( convolution ( d, R ) ) ;
        if( fc>fd ) :
            b , d = d , c
            c = b - golden_ratio * ( b - a )
        else :
            a , c = c , d
            d = a + golden_ratio * ( b - a )
        optimum = 0.5*( c+d )
    return ( optimum )

def isosurface_optimisation():
    print ( "ISOOPT..." )

if __name__ == '__main__':
    data = pd.read_csv( "rich.dat","\t",index_col=0 )
    print ( golden_ration_phasetransition_search ( data.values ) )


    

