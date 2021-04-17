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

import numpy as np
import pandas as pd
import operator

def F ( i , j , d = 2 ,
        s = lambda x,y : 2 * (x==y) ,
        sx = None , sy = None ) :

    if operator.xor( sx is None , sy is None ):
        return ( -1 )
    if i == 0 and j == 0 :
        return ( s(sx[0],sy[0]) )
    if operator.xor( i==0 , j==0 ) :
        return ( -d*(i+j) )
    return ( np.max( [ F( i-1 , j-1 , sx=sx , sy=sy ) + s( sx[i-1],sy[j-1] ) ,
                       F( i-1 , j   , sx=sx , sy=sy ) - d ,
                       F( i   , j-1 , sx=sx , sy=sy ) - d ] ) )

def scoring_function ( l1,l2 ) :
    s_ = np.log2(  2*( l1==l2 ) + 1 )
    return ( s_ )

def check_input ( strp ):
    err_msg = "must be called with two strings placed in a list"
    bad = False
    if not 'list' in str( type(strp) ) :
        bad = True
    else:
        for str_ in strp :
            if not 'str' in str(type(str_)):
                bad=True
    if bad :
        print ( err_msg )
        exit ( 1 )

def sdist ( strp , scoring_function = scoring_function ) :
    check_input( strp )
    s1 , s2 = strp[0] , strp[1]
    N  , M  = len(s1) , len(s2)
    mg = np.meshgrid( range(N),range(M) )
    W  = np.zeros(N*M).reshape(N,M)
    for pos in zip( mg[0].reshape(-1),mg[1].reshape(-1) ):
        pos_ = np.array( [(pos[0]+0.5)/N , (pos[1]+0.5)/M] )
        dij = np.log2( np.sum( np.diff(pos_)**2 ) + 1 ) + 1
        sij = scoring_function( s1[pos[0]],s2[pos[1]] )
        W [ pos[0],pos[1] ] = sij/dij
    return ( W )

def score_alignment ( string_list ,
                      scoring_function = scoring_function ,
                      shift_allowance = 1 , off_diagonal_power=None,
                      main_diagonal_power = 2 ) :
    check_input(string_list)
    strp  = string_list.copy()
    n,m   = len(strp[0]) , len(strp[1])
    shnm  = [n,m]
    nm,mn = np.max( shnm ) , np.min( shnm )
    axis  = int( n>m )
    paddington = np.repeat([s for s in strp[axis]],shnm[axis]).reshape(shnm[axis],shnm[axis]).T.reshape(-1)[:nm]
    strp[axis] = ''.join(paddington)
    W          = sdist( strp , scoring_function=scoring_function)
    if axis==1 :
        W = W.T
    Smax , SL = 0,[0]

    mdp = main_diagonal_power
    sha = shift_allowance
    for i in range(nm) :
        Sma_ = np.sum( np.diag( W,i ))**mdp
        for d in range( sha ) :
            p_ = 1.
            d_ = d + 1
            if 'list' in str(type(off_diagonal_power)):
                if len ( off_diagonal_power ) == sha :
                    p_ = off_diagonal_power[d]
            if i+d_ < nm :
                Sma_ += np.sum( np.diag( W , i+d_ ))**p_
            if i-d_ >= 0 :
                Sma_ += np.sum( np.diag( W , i-d_ ))**p_
        if Sma_ > Smax:
            Smax = Sma_
            SL.append(Sma_)
    return ( Smax/(2*sha+1)/(n+m)*mn )

def read_xyz(name='data/naj.xyz',header=2,sep=' '):
    mol_str = pd.read_csv(name,header=header)
    P=[]
    for i_ in range(len(mol_str.index)):
        line = mol_str.iloc[i_,:].values[0]
        lsp = [l.replace(' ','') for l in line.split(sep) if len(l)>0]
        P.append(lsp)
    pdf = pd.DataFrame(P); pdf.index=pdf.iloc[:,0].values ; pdf=pdf.iloc[:,1:4]
    return(pdf.apply(pd.to_numeric))

def KabschAlignment( P,Q ):
    #
    # https://en.wikipedia.org/wiki/Kabsch_algorithm
    # C++ VERSION: https://github.com/richardtjornhammar/RichTools/blob/master/src/richfit.cc
    #	IN VINCINITY OF LINE 524
    #
    N,DIM  = np.shape( P )
    M,DIM  = np.shape( Q )
    if DIM>N or not N==M :
        print( 'MALFORMED COORDINATE PROBLEM' )
        exit( 1 )

    q0 , p0 = np.mean(Q,0) , np.mean(P,0)
    cQ , cP = Q - q0 , P - p0

    H = np.dot(cP.T,cQ)
    I  = np.eye( DIM )

    U, S, VT = np.linalg.svd( H, full_matrices=False )
    Ut = np.dot( VT.T,U.T )
    I[DIM-1,DIM-1] = 2*(np.linalg.det(Ut) > 0)-1
    ROT = np.dot( VT.T,np.dot(I,U.T) )
    B = np.dot(ROT,P.T).T + q0 - np.dot(ROT,p0)

    return ( B )


def WeightsAndScoresOf( P , bFA=False ) :
	p0 = np.mean( P,0 )
	U, S, VT = np.linalg.svd( P-p0 , full_matrices=False )
	weights = U
	if bFA :
		scores = np.dot(S,VT).T
		return ( weights , scores )
	scores = VT.T
	return ( weights , scores )

def ShapeAlignment( P, Q ,
		 bReturnTransform = False ,
		 bShiftModel = True ,
		 bUnrestricted = False ) :
    #
    # [*] C++ VERSION: https://github.com/richardtjornhammar/RichTools/blob/master/src/richfit.cc
    # FIND SHAPE FIT FOR A SIMILIAR CODE IN THE RICHFIT REPO
    #
    description = """
     A NAIVE SHAPE FIT PROCEDURE TO WHICH MORE SOPHISTICATED
     VERSIONS WRITTEN IN C++ CAN BE FOUND IN MY C++[*] REPO

     HERE WE WORK UNDER THE ASSUMPTION THAT Q IS THE MODEL
     SO THAT WE SHOULD HAVE SIZE Q < SIZE P WITH UNKNOWN
     ORDERING AND THAT THEY SHARE A COMMON SECOND DIMENSION

     IN THIS ROUTINE THE COARSE GRAINED DATA ( THE MODEL ) IS
     MOVED TO FIT THE FINE GRAINED DATA ( THE DATA )
    """

    N,DIM  = np.shape( P )
    M,DIM  = np.shape( Q )
    W = (N<M)*N+(N>=M)*M

    if (DIM>W or N<M) and not bUnrestricted :
        print ( 'MALFORMED PROBLEM' )
        print ( description )
        exit ( 1 )

    q0 , p0 = np.mean(Q,0) , np.mean(P,0)
    cQ , cP = Q - q0 , P - p0
    sQ = np.dot( cQ.T,cQ )
    sP = np.dot( cP.T,cP )

    H = np.dot(sP.T,sQ)
    I = np.eye( DIM )

    U, S, VT = np.linalg.svd( H, full_matrices=False )
    Ut = np.dot( VT.T,U.T )
    I[DIM-1,DIM-1] = 2*(np.linalg.det(Ut) > 0)-1
    ROT = np.dot( VT.T,np.dot(I,U.T) )
    if bReturnTransform :
        return ( ROT,q0,p0 )

    if bShiftModel :# SHIFT THE COARSE GRAINED DATA
        B = np.dot(ROT,Q.T).T +p0 - np.dot(ROT,q0)
    else : # SHIFT THE FINE GRAINED DATA
        B = np.dot(ROT,P.T).T +q0 - np.dot(ROT,p0)

    return ( B )


def low_missing_value_imputation ( fdf , fraction = 0.9 , absolute = 'True' ) :
    # THIS SVD BASED IMPUTATION METHOD WAS FIRST WRITTEN FOR THE RANKOR PACKAGE
    # ORIGINAL CODE IN https://github.com/richardtjornhammar/rankor/blob/master/src/rankor/imputation.py
    #
    import numpy as np
    #
    # fdf is a dataframe with NaN values
    # fraction is the fraction of information that should be kept
    # absolute is used if the data is positive
    #
    V = fdf.apply(pd.to_numeric).fillna(0).values
    u,s,vt = np.linalg.svd(V,full_matrices=False)
    s =  np.array( [ s[i_] if i_<np.floor(len(s)*fraction) else 0 for i_ in range(len(s)) ] )
    nan_values = np.dot(np.dot(u,np.diag(s)),vt)
    if absolute :
        nan_values = np.abs(nan_values)
    #
    # THIS CAN BE DONE BETTER
    for j in range(len(fdf.columns.values)):
        for i in range(len(fdf.index.values)):
            if 'nan' in str(fdf.iloc[i,j]).lower():
                fdf.iloc[i,j] = nan_values[i,j]
    return ( fdf )

if __name__ == '__main__' :
    #
    # IF YOU REQUIRE THE DATA THEN LOOK IN :
    # https://github.com/richardtjornhammar/RichTools
    # WHERE YOU CAN FIND THE FILES USED HERE
    #
    if False :
        colors = {'H':'#777777','C':'#00FF00','N':'#FF00FF','O':'#FF0000','P':'#FAFAFA'}
        Q = read_xyz( name='data/naj.xyz'   , header=2 , sep=' ' )

    if False : # TEST KABSCH ALGORITHM
        P = Q .copy()
        Q = Q * -1
        Q = Q + np.random.rand(Q.size).reshape(np.shape(Q.values))

        P_ , Q_ = P.copy() , Q.copy()
        P = P_.values
        Q = Q_.values
        B = KabschAlignment( P,Q )
        B = pd.DataFrame( B , index = P_.index.values ); print( pd.concat([Q,B],1))

    if False : # TEST MY SHAPE ALGORITHM
        P = read_xyz ( name='data/cluster0.xyz' , header=2 , sep='\t' )
        P_ , Q_= P.values,Q.values
        B_ = ShapeAlignment( P_,Q_ )
        B = pd.DataFrame(B_, index=Q.index,columns=Q.columns)
        pd.concat([B,P],0).to_csv('data/shifted.xyz','\t')

    if True :
        strpl = [ [ 'ROICAND'    , 'RICHARD' ] ,
                  [ 'RICHARD'    , 'RICHARD' ] ,
                  [ 'ARDARDA'    , 'RICHARD' ] ,
                  [ 'ARD'        , 'RICHARD' ] ,
                  [ 'DRA'        , 'RICHARD' ] ,
                  [ 'RICHARD'    , 'ARD'     ] ,
                  [ 'RICHARD'    , 'DRA'     ] ,
                  [ 'ÖoBasdasda' , 'RICHARD' ] ,
                  [ 'Richard'    , 'Ingen äter lika mycket ris som Risard när han är arg och trött']]
        strp = strpl[0]
        W = sdist ( strp )
        for strp in strpl :
            print ( strp , score_alignment( strp , main_diagonal_power=3.5 , shift_allowance=2, off_diagonal_power=[1.5,0.5]) )
            #print ( strp , score_alignment( strp , main_diagonal_power=3.5 , off_diagonal_power=[1.5]) )
            #print ( strp , score_alignment( strp ) )
