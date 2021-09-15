import numpy as np

def sign ( x ) :
    return ( 2*(x>=0)-1 )

def factorial ( n ) :
    return ( 1 if n<=0 else factorial(n-1)*n )

def invfactorial ( n ) :
    if n<0 :
        return 0
    m = factorial(n)
    return ( 1/m )

def zernicke ( r , theta , n , m ) :
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
