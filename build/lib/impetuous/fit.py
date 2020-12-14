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

import numpy as np
import pandas as pd

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
	if True :
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
   
	if True : # TEST MY SHAPE ALGORITHM
            P = read_xyz ( name='data/cluster0.xyz' , header=2 , sep='\t' )
            P_ , Q_= P.values,Q.values
            B_ = ShapeAlignment( P_,Q_ )
            B = pd.DataFrame(B_, index=Q.index,columns=Q.columns)
            pd.concat([B,P],0).to_csv('data/shifted.xyz','\t')



            
