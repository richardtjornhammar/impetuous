"""
Copyright 2023 RICHARD TJÖRNHAMMAR

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
from scipy.stats import kurtosis
from scipy.stats import rankdata

def get_coordinates ( values:np.array , length_scales:list=None ) -> np.array :
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

def fieldform ( values:np.array , length_scales:list=None ) :
    coordinates = get_coordinates( values , length_scales=length_scales )
    R = [ coordinates , values ]
    return ( R )

def equalise_field ( noisy:np.array ) -> np.array :
    # HISTOGRAM EQUALISATION
    return ( accentuate_field ( noisy , False ) )

def accentuate_field ( noisy:np.array , bExp:bool = True ) -> np.array :
    # NOISY IS A TENSOR CONTAINING VALUES DESCRIBING SOME INTERNALLY SIMILAR TYPED OBJECT
    accf  = rankdata(noisy.reshape(-1),'average').reshape(np.shape(noisy))
    accf /= (np.max(accf)+0.5)
    if bExp :
        accf = 2*( np.exp(accf) - 1 )
    return ( accf )

#sigma = 0.5 / np.abs(xi)**0.5
#nrm   = np.sqrt(2.0*np.pi*sigma)**N
def convolve ( xi,R,bFlat = True ) :
    fval = np.fft.fft2( R[1])
    G    = np.exp( -( np.sum( np.array( R[0] )**2 ,0) )*( xi ) )
    conn = np.fft.ifft2(np.fft.fftshift((np.fft.fftshift(fval)*G))).real
    if bFlat :
        conn = conn.reshape(-1)
    return ( conn )


def nn ( i:int , j:int , nnL:int , L:int , P:int) -> list :
    NN = []
    for k in range(i-nnL,i+nnL+1):
        if k<0 or k>L-1:
            continue
        for l in range(j-nnL,j+nnL+1):
            if l<0 or l>P-1:
                continue
            if k==i and l==j:
                continue
            NN.append( (k,l) )
    return ( NN )


def best_neighbor_assignment ( noisy:np.array , nnL:int = 1 ) -> np.array :
    # SLOW METHOD FOR DIRECTIONAL BLURRING USING THE BEST NEIGHBOR VALUE
    # SAME ENTROPY
    res = noisy.copy()
    IDX = [ (ic,jc) for ic in range(res.shape[0]) for jc in range(res.shape[1]) ]
    for idxpair in IDX :
            ic = idxpair[0]
            jc = idxpair[1]
            idxs = nn ( ic , jc , nnL , res.shape[0] , res.shape[1] )
            nnvals = [ noisy[rp] for rp in idxs ]
            armin  = np.argmin( (nnvals - noisy[ic,jc])**2 )
            repval = 0.5*( nnvals[armin] + noisy[ic,jc] )
            for rp in idxs :
                res[rp] += repval / len(idxs)
    return ( res )

def fd001 ( I:np.array , npix:int=3 ) -> np.array :
    # INCREASES ENTROPY
    AI      = accentuate_field ( I )
    N,M     = np.shape(AI)
    DIJ     = I.copy() * 0.0
    PAJ     = np.pad ( AI , npix )*0.0
    for i in range ( -npix , npix ) :
        for j in range ( -npix , npix ) :
            AJ  = AI
            D2  = 0
            if i==0 and j==0 :
                continue
            AJ  = np.roll( AJ , i , axis=0 )
            AJ  = np.roll( AJ , j , axis=1 )
            I2  = ( AJ - AI )**2  / np.std(AI)**2
            D2  = ( i**2 + j**2 ) / npix**2
            mX  = 0.5 * ( I2 + D2 )
            mlX = 0.5 * ( np.log(I2) + np.log(D2) )
            X2  = ( np.exp( mlX ) ) / ( mX )
            PAJ[ (npix+i):(N+npix+i),(npix+j):(M+npix+j) ] += X2
    RAJ = PAJ[npix:-npix,npix:-npix]
    return ( RAJ )

def fdz ( I:np.array , npix:int=5 , cval:float=50 , bEqualized:bool=True ) -> np.array :
    # NOT DIAGNOSED
    AI = I.copy()
    if bEqualized :
        AI      = accentuate_field ( I , False )
    N,M     = np.shape(AI)
    PAJ     = np.pad ( AI , npix ) * 0.0
    for i in range ( -npix , npix ) :
        for j in range ( -npix , npix ) :
            AJ  = AI
            D2  = 0
            if i == 0 and j == 0 :
                continue
            AJ  = np.roll( AJ , i , axis=0 )
            AJ  = np.roll( AJ , j , axis=1 )
            I2  = ( AJ - AI )**2  / np.std(AI)**2
            D2  = ( i**2 + j**2 ) / npix**2
            mX  = 0.5 * ( I2 + D2 )
            mlX = 0.5 * ( np.log(I2) + np.log(D2) )
            X2  = ( np.exp( mlX ) ) / ( mX )
            PAJ [ (npix+i):(N+npix+i),(npix+j):(M+npix+j) ] += X2
    RAJ = PAJ[npix:-npix,npix:-npix]
    return ( field_convolve(RAJ*AI,cval)  )


def field_convolve( values:np.array, mask_value:float=0, convolution = lambda xi,R:convolve(xi,R,False) ) : # LEAVE LAMBDAS UNTYPED FOR NOW
    R = fieldform ( values )
    return ( convolution ( mask_value , R ) )

def golden_ration_phasetransition_search ( values:np.array , coordinates:np.array = None ,
                           unimodal_function = lambda x:kurtosis(x) ,
                           convolution = lambda xi,R:convolve(xi,R) ,
                           length_scales:list = None , extreme:float = 1000.0, tol:float=1e-6 ):
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


def GRS (        data:np.array                                           ,
                 aux_data:np.array       = None                          ,
                 coordinates:np.array    = None                          ,
                 unimodal_function       = lambda x:kurtosis(x)          ,
                 transform               = lambda xi,R:convolve(xi,R)    ,
                 length_scales:list      = None                          ,
                 extremes:list           = None                          ,
                 power:float             = 2                             ,
                 tol:float               = 1e-6                          ) -> float :
    #
    information__ = """
Adapted from impetuous.optimisation import golden_ration_phasetransition_search
see my github.com/richardtjornhammar/MapTool repo i.e. file
maptool.cc  around line 845 or
maptool2.cc around line 646
"""
    R = [ data ]
    if coordinates is None and not length_scales is None :
        coordinates = get_coordinates(data,length_scales)
        R = [ coordinates,*R ]

    if not aux_data is None :
        R = [ *R , aux_data ]
    #
    golden_ratio    = ( 5.0**0.5-1.0 )*0.5
    a , b , fa, fb , fc , fd = np.min(np.min(R[0])), np.max(np.max(R[0])), 0.0, 0.0, 0.0, 0.0
    if not extremes is None :
        a = extremes[0]
        b = extremes[1]

    c , d =  b-golden_ratio*(b-a), a+golden_ratio*(b-a)
    metric, optimum = 0.0 , 0.0
    while ( d-c > tol ) :
        fc = ( unimodal_function( transform ( c, R ) ) )**power
        fd = ( unimodal_function( transform ( d, R ) ) )**power
        if( fc >= fd or fd == fb ) :
            b , d = d , c
            c = b - golden_ratio * ( b - a )
        else :
            a , c = c , d
            d = a + golden_ratio * ( b - a )
        fa , fb = fc , fd
        optimum = 0.5 * ( c+d )
    return ( optimum )

if __name__ == '__main__':
    data = pd.read_csv( "rich.dat","\t",index_col=0 )
    print ( golden_ration_phasetransition_search ( data.values ) )




