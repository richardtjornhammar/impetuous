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

desc__ = """

THIS MODULE WILL CONTAIN SINGLE FUNCTIONS THAT HAVE NO OTHER DEPENDENCIES
EXCEPT NUMPY. SOME HAVE BEEN MOVED OVER FROM MODULES WITH COMPLICATED
DEPENDENCY STRUCTURES AND WILL ALLEVIATE POTENTIALLY DIFFICULT GRAPH LOOPS
AS WELL AS EASE DEBUGGING IN THE FUTURE.

CURRENT: REDUNDANT DEFINITIONS (HERE VS REDUCER) EXISTS FOR

reducer::
frac_procentile
get_procentile
smoothbinred
smoothmax
contrast
e_flatness
e_contrast
confred
padded_rolling_window

quantification::
isItPrime
# FIRST APPEARENCE:
# https://gist.github.com/richardtjornhammar/ef1719ab0dc683c69d5a864cb05c5a90
Fibonacci
F_truth
"""

import numpy as np

def sign ( x ) :
    return ( 2*(x>=0)-1 )

def abs ( x ):
    return(sign(x)*x)

contrast   = lambda A,B : ( A-B )/( A+B )
#
# OTHER
e_flatness = lambda x   : np.exp(np.mean(np.log(x),0))/np.mean(x,0)
e_contrast = lambda x   : 1 - e_flatness(x)
#
# SPURIOUS LOW VALUE REMOVAL
confred       = lambda x,eta,varpi : 0.5*x*(1+np.tanh((x-eta)/varpi))*(np.sqrt(x*eta)/(0.5*(eta+x)))
smoothbinred  = lambda x,eta,varpi : 0.5*(1+np.tanh((x-eta)/varpi))*(np.sqrt(x*eta)/(0.5*(eta+x)))
smoothmax     = lambda x,eta,varpi : x * self.smoothbinred(x-np.min(x),eta-np.min(x),varpi)
sabsmax       = lambda x,eta,varpi : x * self.smoothbinred(np.abs(x),np.abs(eta),varpi)

def frac_procentile ( vals=[12.123, 1.2, 1000, 4] ):
    vals = np.array(vals).copy()
    N = len( vals );
    for i,j in zip( np.argsort(vals),range(N)):
        vals[i]=(j+0.5)/N
    return ( vals )

def get_procentile ( vals, procentile = 50 ):
    fp_   = procentile/100.0
    proc  = frac_procentile(vals)
    ma,mi = np.max(proc),np.min(proc)
    if fp_ > ma:
        fp_ = ma
    if fp_ < mi:
        fp_ = mi
    idx = np.argwhere(proc==fp_)
    if len ( idx ) == 1 :
        return ( vals[idx[0][0]] )
    else :
        i1 = np.argsort( np.abs(proc-fp_) )[:2]
        return ( sum([vals[i] for i in i1]) * 0.5 )

def padded_rolling_window ( ran, tau ) :
        if tau==1 :
                return ( [ (i,None) for i in range(len(ran)) ] )
        if len(ran)<tau :
                return ( [ (0,N) for v_ in ran ] )
        centered = lambda x:(x[0],x[1]) ; N=len(ran);
        w = int( np.floor(np.abs(tau)*0.5) ) ;
        jid = lambda i,w,N:[int((i-w)>0)*((i-w)%N),int(i+w<N)*((i+w)%N)+int(i+w>=N)*(N-1)]
        idx = [ centered( jid(i,w,N) ) for i in range(N) ]
        return ( idx )

def factorial ( n ) :
    return ( 1 if n<=0 else factorial(n-1)*n )

def invfactorial ( n ) :
    if n<0 :
        return 0
    m = factorial(n)
    return ( 1/m )

def zernicke ( r , theta , n , m ) :
    print ( 'WARNING: THIS FUNCTION IS STILL UNDER EVALUATION' )
    if ( not (r >= 0 and r <= 1)) or (m > n) :
        return ( 0 )

    def zer_R ( n , m , r ) :
        ip,im = ( n+m )/2 , ( n-m )/2
        z = 0
        for k in range( int( im ) ) :
            f = factorial(n-k)*invfactorial(k)*invfactorial(ip-k)*invfactorial(im-k)
            if f > 0 :
                z = z + (-1)**k * f * r**( n-2*k )
        return ( z )

    Rnm  = zer_R ( n,m,r )
    Zeve = Rnm * np.cos ( m*theta )
    Zodd = Rnm * np.sin ( m*theta )

    return ( [ Zeve,Zodd ] )

def error ( self , errstr , severity=0 ):
    print ( errstr )
    if severity > 0 :
        exit(1)
    else :
        return

def seqsum ( c , n = 1 ) :
    return ( c[n:] + c[:-n] )

def seqdiff ( c , n = 1 ) :
    return ( c[n:] - c[:-n] )

def arr_contrast ( c , n=1, ctyp='c' ) :
    if ctyp=='c':
        return ( np.append( (c[n:]-c[:-n])/(c[n:]+c[:-n]) ,  np.zeros(n)  ) )
    return ( np.append( np.zeros(n) , (c[n:]-c[:-n])/(c[n:]+c[:-n]) ) )

def all_conts ( c ) :
    s   = c*0.0
    inv = 1.0/len(c)
    for i in range(len(c)):
        s += arr_contrast(c,i+1)*inv
    return ( s )

def mse ( Fs,Fe ) :
    return ( np.mean( (Fs-Fe)**2 ) )

def coserr ( Fe , Fs ) :
    return ( np.dot( Fe,Fs )/np.sqrt(np.dot( Fe,Fe ))/np.sqrt(np.dot( Fs,Fs )) )

def z2error ( model_data , evidence_data , evidence_uncertainties = None ) :
    Fe = evidence_data
    Fs = model_data
    N  = np.min( [ len(evidence_data) , len(model_data) ] )
    if not  len(evidence_data) == len(model_data):
        error ( " THE MODEL AND THE EVIDENCE MUST BE PAIRED ON THEIR INDICES" , 0 )
        Fe  = evidence_data[:N]
        Fs  = model_data[:N]

    dFe = np.array( [ 0.05 for d in range(N) ] )
    if not evidence_uncertainties is None :
        if len(evidence_uncertainties)==N :
            dFe = evidence_uncertainties
        else :
            error ( " DATA UNCERTANTIES MUST CORRESPOND TO THE TARGET DATA " , 0 )

    def K ( Fs , Fe , dFe ) :
        return ( np.sum( np.abs(Fs)*np.abs(Fe)/dFe**2 ) / np.sum( (Fe/dFe)**2 ) )

    k = K ( Fs,Fe,dFe )
    z2e = np.sqrt(  1/(N-1) * np.sum( ( (np.abs(Fs) - k*np.abs(Fe))/(k*dFe) )**2 )  )
    cer = coserr(Fe,Fs)
    qer = z2e/cer

    return ( qer, z2e , cer , N )

import math
def isItPrime( N , M=None,p=None,lM05=None ) :
    if p is None :
        p = 1
    if M is None :
        M = N
    if lM05 is None:
        lM05 = math.log(M)*0.5
    if ((M%p)==0 and p>=2) :
        return ( N==2 )
    else :
       if math.log(p) > lM05:
           return ( True )
       return ( isItPrime(N-1,M=M,p=p+1,lM05=lM05) )

# FIRST APPEARENCE:
# https://gist.github.com/richardtjornhammar/ef1719ab0dc683c69d5a864cb05c5a90
def Fibonacci(n):
    if n-2>0:
        return ( Fibonacci(n-1)+Fibonacci(n-2) )
    if n-1>0:
        return ( Fibonacci(n-1) )
    if n>0:
       return ( n )

def F_truth(i):
    return ( Fibonacci(i)**2+Fibonacci(i+1)**2 == Fibonacci(2*i+1))

if __name__ == '__main__' :
    print ( sign(-10) )
    print (  abs(-10) , abs(10) )
    print ( factorial ( 3 ) )
    print ( factorial ( 0 ) )
    print ( factorial (-3 ) )

    x = np.array( [ a for a in range(20) ] )
    Y = all_conts(x)
    Y = all_conts(1/x)

    Y = all_conts( np.exp( -((x-5)/3)**2 )  )


    print ( arr_contrast(x) )
    print ( seqdiff(x) )
    print ( np.diff(x) )
    print ( seqsum(x) )

    import matplotlib.pyplot as plt
    plt.figure(1).clear()
    plt.plot(x,-Y, 'b'  )
    plt.show()
