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

import numpy as np
import pandas as pd
import operator
import typing

class quaternion ( ) :
    def __init__ ( self , vector=None , angle=None ):
        self.bComplete = False
        self.v         = vector
        self.angle     = angle
        self.q         = np.array([0.,0.,0.,0.])
        self.qrot      = None
        self.assign_quaternion()

    def __eq__  ( self , other ) :
        return ( True )

    def __str__ ( self ) :
        return ( self.info() )

    def __repr__( self ) :
        return ( self.info() )

    def info( self ):
        desc__= """< quaternion > instance at address [ """ + hex(id(self)) + """ ]\n""" + \
                """  quaternion > """ + ', '.join( [ str(v_) for v_ in self.q ] ) + \
                """ \n  | angle  = """ + str ( self.angle ) + \
                """ \n  | vector = """ + ', '.join( [ str(v_) for v_ in self.v ] )
        return ( desc__ )

    def get( self ) :
        return ( [ self.U, self.S, self.VT ] )

    def assign_quaternion (self ,  v=None , angle=None ):
        if v is None :
            v = self.v
        else :
            self.v = v
        if angle is None :
            angle = self.angle
        else :
            self.angle = angle
        if angle is None or v is None :
            self.bComplete = False
            return
        else :
            self.bComplete = True
        fi = angle*0.5
        norm = 1.0 / np.sqrt( np.sum( v**2 ) )
        self.q[0] = np.cos(fi)
        self.q[1] = v[0]*norm*np.sin(fi)
        self.q[2] = v[1]*norm*np.sin(fi)
        self.q[3] = v[2]*norm*np.sin(fi)
        self.calc_rotation_matrix()

    def calc_rotation_matrix(self):
        if self.bComplete :
            q = self.q
            self.qrot = np.array( [ [ q[0]*q[0]+q[1]*q[1]-q[2]*q[2]-q[3]*q[3] , 2*q[1]*q[2] - 2*q[0]*q[3] , 2*q[1]*q[3] + 2*q[0]*q[2] ] ,
                                    [ 2*q[1]*q[2] + 2*q[0]*q[3] , q[0]*q[0]-q[1]*q[1] + q[2]*q[2]-q[3]*q[3] , 2*q[2]*q[3]-2*q[0]*q[1] ] ,
                                    [ 2*q[1]*q[3] - 2*q[0]*q[2] , 2*q[2]*q[3] + 2*q[0]*q[1] , q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3] ] ] )
    def rotate_coord (self, x ) :
        if self.bComplete :
            return ( np.dot(self.qrot,x) )

class NonSequentialReservoirComputing ( ) :
    def __init__ ( self ,
            data           = None  ,
            data_length    = None  ,
            reservoir_size = None  ,
            leak_factor    = 0.3   ,
            alpha          = 1E-8  ,
            bSequential    = False ,
            nmem           = 2     ,
            seed_id        = 11111 ) :

        self.smoothbinred  = lambda x,eta,varpi : 0.5*(1+np.tanh((x-eta)/varpi))*(np.sqrt(x*eta)/(0.5*(eta+x)))
        self.smoothmax     = lambda x,eta,varpi : x * self.smoothbinred(x-np.min(x),eta-np.min(x),varpi)
        self.sabsmax       = lambda x,eta,varpi : x * self.smoothbinred(np.abs(x),eta,varpi)

        self.indata        = data
        self.target        = data
        self.data_length   = data_length
        if ( not data is None ) :
            nm = np.shape(data)
            if len(nm) > 1 :
                if nm[0] < nm[1] :
                    data = data.T
                self.target = data.T[0]
                if nm[0]>1 :
                    self.target  = data.T[1]
                self.indata  = data.T[0]
            self.data_length = len ( data )

        if data_length is None :
            self.data_length = len( self.indata )

        self.leak_factor     = leak_factor
        self.alpha           = alpha
        self.seed_id         = seed_id
        self.coldim          = 1
        self.nres            = reservoir_size
        if reservoir_size is None :
            self.nres        = int( np.ceil(self.data_length*0.2) )

        self.W        = None # A POOL OF SAIGA NEURONS
        self.Win      = None
        self.Wout     = None
        self.X        = [None,None]
        self.pathways = [None,None]
        self.Y        = None
        self.Yt       = None
        self.z2err    = None

        self.nmem     = nmem
        if self.nres <= self.nmem :
            self.nmem = self.nres-1

        self.bSequential = bSequential

        if not self.indata is None :
            self.init()
            self.train()
            self.generate()
            self.z2error()

    def __eq__  ( self , other ) :
        return ( True )

    def __str__ ( self ) :
        return ( self.info() + '\n\n' + str( self.get() ) )

    def __repr__( self ) :
        return ( self.get() )


    def info( self ) :
        desc__ = """
!!! YOU MIGHT STILL BE RUNNING A SEQUENTIALLY DEPENDENT ANALOG!!!
NON SEQUENTIAL RESERVOIR COMPUTING
IMPLEMENTED FOR TESTING PURPOSES : DEVELOPMENTAL
        """
        return ( desc__ )

    def init ( self ) :
        np.random.seed( self.seed_id )
        self.Win  = np.random.rand( self.nres , self.nmem + self.coldim ) - 0.5
        self.W    = np.random.rand( self.nres ,     self.nres   ) - 0.5
        self.W   /= np.sum( np.diag(self.W) )

    def stimulate_neurons ( self , indat , io=0 , bSVD=False , bSequential=False ) :
        n        = len (  indat )
        nres     = len ( self.W )
        indat0   = np.array( [ i_ for i_ in indat] )
        if bSVD and io == 0 :
            Y_   = np.dot( self.Win, np.vstack( ( np.ones(n*(self.nmem-1)).reshape((self.nmem-1),n),self.target,indat0) )  )
            self .Win = np.linalg.svd(Y_)[0][:,:(self.nmem + self.coldim)]

        indat_   = np.dot ( self.Win , np.vstack( (np.ones(n*(self.nmem)).reshape((self.nmem),n),indat0) ) )

        if bSequential :
            x = np.zeros((nres,1)) ; X = []
            for i in range( n ) :
                xp = (indat_.T[i,:] + np.dot( self.W , x ).T[0]).reshape(-1,1)
                x  = (1-self.leak_factor)* x + self.leak_factor*np.tanh(xp)
                X.append( x.reshape(-1) )
            X = np.array( X ).T
        else :
            xi = np.dot( self.W , indat_ )
            X  = self.sabsmax( xi , np.sqrt( np.mean(xi**2) ) , np.mean(xi**2) )

        if io == 0 :
            Yt = self.target
            self.Wout = np.linalg.solve( np.dot(X,X.T) + self.alpha*np.eye(nres) , np.dot( X , Yt ))
        if io == 1 :
            self.Y = np.dot ( self.Wout, X )
        self.X[io] = X
        return

    def train( self , data = None ) :
        if data is None :
            data = self.indata
        self.stimulate_neurons( data , io=0 , bSequential=self.bSequential )
        return

    def generate ( self , userdata = None ) :
        if not userdata is None :
            self.stimulus = userdata
        else :
            self.stimulus = self.indata
        self.stimulate_neurons( self.stimulus , io=1 , bSequential=self.bSequential )
        self.z2error()
        return self.Y

    def error ( self , errstr , severity = 0 ):
        print ( errstr )
        if severity > 0 :
            exit(1)
        else :
            return

    def coserr ( self, Fe , Fs ) :
        return ( np.dot( Fe,Fs )/np.sqrt(np.dot( Fe,Fe ))/np.sqrt(np.dot( Fs,Fs )) )

    def z2error ( self, data_uncertanties = None ) :
        N   = np.min( [ len(self.target) , len(self.Y) ] )
        Fe  = self.target[:N]
        Fs  = self.Y[:N]
        if data_uncertanties is None :
            dFe = np.array( [ 0.05 for d in range(N) ] )
        else :
            if len(data_uncertanties)<N :
                self.error( " DATA UNCERTANTIES MUST CORRESPOND TO THE TARGET DATA " ,0 )
            dFe = data_uncertanties[:N]
        def K ( Fs , Fe , dFe ) :
            return ( np.sum( np.abs(Fs)*np.abs(Fe)/dFe**2 ) / np.sum( (Fe/dFe)**2 ) )
        k = K ( Fs,Fe,dFe )
        z2e = np.sqrt(  1/(N-1) * np.sum( ( (np.abs(Fs) - k*np.abs(Fe))/(k*dFe) )**2 )  )
        cer = self.coserr(Fe,Fs)
        qer = z2e/cer
        self.z2err = ( qer, z2e , cer , self.nres, N )
        return

    def get ( self ) :
        return ( { 'target data'           : self.target   ,
                   'predicted data'        : self.Y        ,
                   'reservoir activations' : self.X        ,
                   'reservoir'             : self.W        ,
                   'output weights'        : self.Wout     ,
                   'input weights'         : self.Win      ,
                   'error estimates'       : self.z2err    } )

class ReservoirComputing ( ) :
    def __init__ ( self ,
            data           = None  ,
            order          = None  ,
            data_length    = None  ,
            work_fraction  = 0.1   ,
            work_length    = None  ,
            train_length   = None  ,
            test_length    = None  ,
            init_length    = None  ,
            input_size     = 1     ,
            output_size    = 1     ,
            reservoir_size = None  ,
            leak_factor    = 0.3   ,
            regularisation_strength = 1E-8,
            type           = 'ESN' ,
            seed_id        = 11111 ,
            bSVD           = True ) :

        self.data        = data
        if ( not data is None ) and ( data_length is None ) :
            nm = np.shape(data)
            if len(nm) > 1 :
                if nm[0] < nm[1] :
                    data = data.T
                data = data.T[0]
                self.data = data
            self.data_length = len ( data )
        self.work_fraction   = work_fraction

        self.assign_dimensions( train_length , test_length , init_length ,
                                input_size   , output_size , reservoir_size ,
                                work_length  , data_length )

        self.input_size   = input_size
        self.output_size  = output_size
        self.leak_factor  = leak_factor
        self.regularisation_strength = regularisation_strength
        self.type         = type

        self.seed_id        = seed_id
        self.bHasAggregates = False
        self.bSVD = bSVD
        self.u    = None
        self.x    = None
        self.X    = None
        self.Y    = None
        self.Yt   = None
        self.Win  = None
        self.Wout = None
        self.rhoW = None

        if not self.data is None :
            self.init( self.bSVD )
            self.train()
            self.generate()
            #self.calc_error()

    def __eq__  ( self , other ) :
        return ( True )

    def __str__ ( self ) :
        return ( self.info() )

    def __repr__( self ) :
        return ( self.info() )

    def info( self ) :
        desc__ = """
 BASIC PRINCIPLES CAN BE STUDIED IN:
 BASED ON        : https://mantas.info/code/simple_esn/ WITH LICENSE https://opensource.org/licenses/MIT
 PUBLICATION     : Harnessing Nonlinearity: Predicting Chaotic Systems and Saving Energy in Wireless Communication
                   Herbert Jaeger and Harald Haas
                   2 APRIL 2004 VOL 304 SCIENCE
 SHOULD BE CITED IF IT IS USED IN SCIENTIFIC WORK.
 THIS IS NOT A COPY. THE ALGORITHM DIFFERS ON SOME POINTS
 BOTH ALGORITHMICALLY AND COMPUTATIONALLY, BUT THE GENERAL IDEA
 WAS ALREADY PRESENTED IN THE ABOVE PAPER
 REMEMBER: YOU CAN FIT ANYTHING ONTO ANYTHING
INPUT PARAMETERS :
            data          
            data_length   
            work_fraction 
            work_length   
            train_length  
            test_length   
            init_length   
            input_size    
            output_size   
            reservoir_size
            leak_factor   
            regularisation_strength
            type          
            seed_id       
            bSVD          
        """
        return ( desc__ )

    def assign_dimensions ( self ,
        train_length = None ,
        test_length  = None ,
        init_length  = None ,
        input_size   = None ,
        output_size  = None ,
        result_size  = None ,
        work_length  = None ,
        data_length  = None ):

        self.data_length = data_length
        if not self.data is None :
            self.data_length = len ( self.data )
        if not self.work_fraction is None :
            work_fraction = self.work_fraction
        if not ( work_fraction>0 and 2*work_fraction<1 ) :
            work_fraction = 0.1
        self.work_fraction = work_fraction
        self.work_length = work_length
        if work_length is None :
            self.work_length = int( np.ceil( self.data_length*work_fraction ) )
        self.train_length = train_length
        if self.train_length is None :
            self.train_length = self.work_length*2
        self.test_length = test_length
        if self.test_length is None :
            self.test_length = self.work_length*2
        self.init_length = init_length
        if self.init_length is None :
            self.init_length = int( np.ceil( self.work_fraction**2*len(self.data) ) )
        self.result_size = result_size
        if result_size is None :
            self.result_size = self.work_length

    def init ( self , bSVD=True ):
        np.random.seed( self.seed_id )
        #
        # result_size IS SET BY reservoir_size INPUT
        self.Win = (np.random.rand( self.result_size, 1+self.input_size ) - 0.5) * 1
        self.W   =  np.random.rand( self.result_size,  self.result_size ) - 0.5
        #
        if bSVD :
            self.rhoW = np.linalg.svd( self.W )[1][0]
        else :
            self.rhoW = np.sum( np.diag(self.W) )
        #
        # RESERVOIR MATRIX
        self.W   *= 1.25 / self.rhoW
        #
        # COLLECTED STATE ACTIVATION MATRIX
        self.X = np.zeros((1+self.input_size+self.result_size,self.train_length-self.init_length))
        self.bHasAggregate = False

    def train ( self , data = None ) :
        #
        self.init( )
        #
        # Y TRAIN
        self.Yt = self.data[ self.init_length+1:self.train_length+1 ]
        #
        # AGGREGATE RESERVOIR, COMPUTE X ACTIVATIONS
        self.x  = np.zeros((self.result_size,1))
        #
        a = self.leak_factor
        for t in range ( self.train_length ) :
            self.u = self.data[t]
            v      =  np.dot( self.Win, np.vstack((1,self.u)) ) + np.dot( self.W, self.x )
            self.x = (1-a)*self.x + a * np.tanh( v )
            if t >= self.init_length :
                self.X[ :,t - self.init_length ] = np.vstack((1,self.u,self.x))[:,0]
        #
        self.u    = self.data[ self.train_length ]
        self.Wout = np.linalg.solve( np.dot ( self.X , self.X.T  ) + \
                             self.regularisation_strength*np.eye(1+self.input_size+self.result_size) ,
                             np.dot ( self.X , self.Yt.T ) ).T
        self.bHasAggregate = True
        self.target = self.data[ self.train_length+1:self.train_length+self.test_length+1 ]
        return

    def generate ( self , start_value = None, nsteps=None ) :
        if not self.bHasAggregate :
            return
        a = self.leak_factor

        if 'float' in str(type(start_value)) :
            self.u = start_value
        if 'int' in str(type(nsteps)) :
            self.test_length = nsteps

        self.Y = np.zeros( (self.output_size,self.test_length) )
        for t in range(self.test_length) :
            v =  np.dot( self.Win, np.vstack((1,self.u)) ) + np.dot( self.W, self.x )
            self.x = (1-a)*self.x + a * np.tanh( v )
            self.y = np.dot( self.Wout, np.vstack((1,self.u,self.x)) )
            self.Y[:,t] = self.y
            self.u = self.y
        return

    def error( self , errstr , severity=0 ):
        print ( errstr )
        if severity > 0 :
            exit(1)
        else :
            return

    def calc_error ( self ) :
        self.error_length = 5*(self.init_length+1)
        if self.train_length+self.error_length+1 >self.data_length:
             self.error ( "BAD LENGTHS>" + str(self.error_length) + " " + str(self.data_length) )
        self.mse = sum( np.square( self.data[self.train_length+1:self.train_length+self.error_length+1] -
                          self.Y[0,0:self.error_length] ) ) / self.error_length
        return

    def get ( self ) :
        return ( { 'target data'           : self.target   ,
                   'predicted data'        : self.Y.T      ,
                   'reservoir activations' : self.X.T      ,
                   'reservoir'             : self.W        ,
                   'output weights'        : self.Wout.T   ,
                   'input weights'         : self.Win.T    } )


class WildDjungle ( object ) :
    def __init__ ( self , distance_type:str = 'euclidean' , bRegressor:bool=False , bReturnDictionaries:bool=False ) :
        self.id_          :str	= "https://github.com/rictjo/biocarta/commit/20a214c253d00903c83d6eb2fa35fb9762eed63f"
        self.description_ :str	= """A WRAPPER CLASS FOR A RANDOM FORERST CLASSSIFIER BUT FIRST EXPANDS MEASURES INTO DISTANCES"""
        self.model_label_ :str	= ""
        self.bRegressor:bool	= bRegressor
        self.bSimplePredict:bool = not bReturnDictionaries
        self.model_order_ :list	= None
        self.array_order_ :list	= None
        self.data_model_df	: pd.DataFrame	= None
        self.target_model_df	: pd.DataFrame	= None
        self.auxiliary_label_df	: pd.DataFrame	= None
        self.bDataCompleted	: bool	= False
        self.descriptive_df	: pd.DataFrame	= None
        self.bg_label_		: str	= 'Background'
        self.fpr_ : np.array	= None
        self.tpr_ : np.array	= None
        self.auc_ : float	= None
        self.cvs_ : np.array	= None
        self.bDidFit_:bool	= False
        import scipy.stats as scs
        from scipy.spatial.distance	import pdist
        from sklearn			import metrics
        from sklearn.ensemble		import RandomForestClassifier
        if self.bRegressor :
            from sklearn.ensemble       import RandomForestRegressor
        self .metrics		= metrics
        self .distance_type:str	= distance_type
        self .pdist		= lambda x : pdist(x,self.distance_type)
        self .RFC		= RandomForestClassifier
        if self.bRegressor :
            self.RFC		= RandomForestRegressor
        self .edge_importances_:np.array	= None
        self .edge_labels_:list[str]		= None
        self .computational_model_		= None # THE UNDERLYING CLASSIFIER MODEL
        self .model_funx	= tuple( ( lambda x:x/np.max(x) , lambda x: np.log(x+1) ) )
        self .func	= lambda x: np.sum(x)
        self .moms	= lambda x: np.array( [np.mean(x),np.std(x),scs.skew(x),-scs.kurtosis(x) , (np.mean(x)*np.std(x))**(1/3) ] )

    def synthesize_data_model ( self , selection:list , adf:pd.DataFrame , jdf:pd.DataFrame , model_label:str , alignment_label:str ) :
        self .model_label_	= model_label
        synth , descriptive	= [] , []
        #
        bdf			= adf.loc[ selection , : ]
        lookupi			= { s:i for s,i in zip(	selection	, range(len(selection)) ) }
        self.model_order_	= sorted ( selection )
        self.array_order_	= [ lookupi[ self.model_order_[k] ] for k in range(len(selection)) ]
        Nr			= int( len(self.model_order_)*(len ( self.model_order_ )-1)*0.5 )
        for c in adf.columns.values :
            model   = bdf.loc[ self.model_order_ , [c] ].apply(pd.to_numeric)
            dists   = self.synthesize( model.values )
            mos     = self.moms( dists )
            descriptive .append( tuple( (jdf.loc[alignment_label,c] if self.model_label_ in jdf.loc[alignment_label,c] else self.bg_label_,
					 self.func( dists ) , *self.moms(dists) )) )
            synth .append( tuple( (jdf.loc[alignment_label,c] if model_label in jdf.loc[alignment_label,c] else self.bg_label_ , *list( dists )  )) )
        res_df = pd.DataFrame( descriptive , columns=['L','V','M1','M2','M3','M4','MM'] )
        self .data_model_df	= pd.DataFrame(synth).iloc[:,1:1+Nr]
        self .target_model_df	= pd.DataFrame(synth).iloc[:,[0]]
        self .descriptive_df	= res_df
        self .bDataCompleted	= True
        self .bDidFit_		= False

    def synthesize ( self , absolute:np.array ) -> np.array :
        return ( self.model_funx[1]( self.pdist( self.model_funx[0](absolute) ) ) )

    def set_model_label ( self, model_label:str ) :
        self.model_label_ = model_label

    def set_bg_label ( self, bg_label:str) :
        self.bg_label_ = bg_label

    def get_model_name(self)->str:
        return ( str(self.model_label_) + ' | ' + str(self.bg_label_) )

    def fit ( self , X:np.array = None , y:np.array = None , binlabel:int = 1 , vertex_labels:list[str] = None ) :
        labels = vertex_labels
        if self.bDataCompleted and (X is None or y is None) :
            ''
        elif not X is None and not y is None :
            self.model_order_ = [ i for i in range(len(X)) ]
            R = []
            for j in range( np.shape(X)[1] ) :
                Z = self.synthesize ( X[:,j].reshape(-1,1) )
                R .append( Z )
            self.data_model_df = pd.DataFrame(R)
            bDone = False
            if self.bRegressor == False :
                if not self.model_label_ is None :
                    if self.model_label_ in set( y ) :
                        v = [ self.model_label_ if self.model_label_ in y_ else self.bg_label_ for y_ in y ]
                        bDone = True
                if not bDone :
                    if binlabel in set(y):
                        v = np.array( [ int(y_ == binlabel) for y_ in y ] )
                    else :
                        print ( 'PLEASE SPECIFY A USEFUL MDOEL LABEL USING .set_model_label(model_label:str) PRIOR TO RUNNING')
                        self.bDidFit_ = False
                        self.bDataCompleted = False
                        exit(1)
                    self.model_label_ = binlabel
            else :
                    v = np.array(y).reshape(-1)
            self.target_model_df	= pd.DataFrame( v )
            self.bDataCompleted = True
        else :
            self.bDataCompleted = False
            print ( 'HAS NO MODEL. RETRAIN THE CLASSIFIER WITH VIABLE INPUT' )
        self .computational_model_	= self.RFC( )
        self .computational_model_ .fit( X=self.data_model_df.values , y=self.target_model_df.values.reshape(-1) )
        self .bDidFit_ = True
        self .edge_importances_ = self .computational_model_.feature_importances_
        nL = np.shape(self.data_model_df.values)[1]
        print ( nL , np.shape(self.data_model_df.values) , np.shape(self.target_model_df.values.reshape(-1)) )
        if labels is None :
            labels = [ str(i+1) for i in range(nL) ]
        else :
            nL = len( labels )
        self.edge_labels_ = [ labels[i]+':'+labels[j]  for i in range(nL) for j in range(nL) if (i<=j and i!=j) ]

    def predict_single_ (self,Y) -> list :
        xvs_ = self.synthesize( Y.reshape(-1,1) ).reshape(1,-1)
        return ( [ {'infered'       : self.computational_model_.predict( xvs_ )[0] ,
                    'probabilities' : self.computational_model_.predict_proba( xvs_ )[0] } ] )

    def predict ( self , X ) -> list :
        if not self.bDidFit_ :
            self.fit()
        nm = np.shape( X )
        if 'panda' in str(type(X)).lower() or 'serie' in str(type(X)).lower() :
            if not self.model_order_ is None :
                if len(set(self.model_order_) - set(X.index.values.tolist()) ) == 0 :
                    X = X .loc[self.model_order_]
                elif np.isreal( np.sum(self.model_order_) ) :
                    X = X.iloc[self.model_order_]
            X = X.values
        elif 'array' in str(type(X)).lower() and not self.array_order_ is None and len(nm)>1 :
            X = X[ self.array_order_ ,: ]
        if len( nm ) > 1 :
            return ( [ self.predict(x_)[0] for x_ in X.T ] )
        else : # np.reshape( np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) ,newshape=(3,3) )
            xvs_ = self.synthesize( X.reshape(-1,1) ).reshape(1,-1)
            if self.bSimplePredict :
                return ( [ self.computational_model_.predict( xvs_ )[0] ] )
            if self.bRegressor :
                return ( [ { 'infered'          : self.computational_model_.predict( xvs_ )[0] } ] )
            else :
                return ( [ { 'infered'		: self.computational_model_.predict( xvs_ )[0] ,
			 'probabilities'	: self.computational_model_.predict_proba( xvs_ )[0] } ] )

    def generate_metrics ( self , y_true:list , y_proba:list , ipos:int= -1 , n_cv:int=5 ) -> dict :
        if y_proba is None or len(y_true) != len(y_proba) :
            print ( "UNEQUAL INPUT LENGTHS NOT SUPPORTED" )
        if 'panda' in str( type(y_true) ) .lower() :
            y_true = [ int(  str(self.model_label_) in str(y)  ) for y in y_true.values.reshape(-1) ]
        if len( y_proba[0] ) == len( y_proba[-1] ) and len( y_proba[0] )>1 :
            y_proba = [ y[ipos] for y in y_proba ]
        fpr_ , tpr_ , rest = self.metrics.roc_curve( y_true , y_proba )
        self .fpr_	= fpr_
        self .tpr_	= tpr_
        self .auc_	= np.trapz( tpr_,fpr_ )
        if not n_cv is None or not self.bDataCompleted :
            from sklearn.model_selection import cross_val_score
            self .cvs_ = cross_val_score(self.computational_model_,self.data_model_df,self.target_model_df.values.reshape(-1),cv=n_cv)
        else :
            self .cvs_ = np.array( ['Not evaluated'] , np.dobject )
        return ( {	'FPR'		: self.fpr_	, 'TPR'	: self.tpr_	,
			'Additonal'	: rest		, 'AUC'	: self.auc_ 	,
			'CV'		: self.cvs_ 	} )


class DjungleClassifier ( WildDjungle ) :
    pass

    def __init__ ( self , distance_type:str = 'euclidean' , bReturnDictionaries:bool=False ) :
        self.id_          :str  = ""
        self.description_ :str  = """A WRAPPER CLASS FOR A RANDOM FORERST CLASSSIFIER BUT FIRST EXPANDS MEASURES INTO DISTANCES"""
        self.model_label_ :str  = ""
        self.bRegressor:bool    = False
        self.bSimplePredict:bool = not bReturnDictionaries
        self.model_order_ :list = None
        self.array_order_ :list = None
        self.data_model_df      : pd.DataFrame  = None
        self.target_model_df    : pd.DataFrame  = None
        self.auxiliary_label_df : pd.DataFrame  = None
        self.bDataCompleted     : bool  = False
        self.descriptive_df     : pd.DataFrame  = None
        self.bg_label_          : str   = 'Background'
        self.fpr_ : np.array    = None
        self.tpr_ : np.array    = None
        self.auc_ : float       = None
        self.cvs_ : np.array    = None
        self.bDidFit_:bool      = False
        import scipy.stats as scs
        from scipy.spatial.distance     import pdist
        from sklearn                    import metrics
        from sklearn.ensemble           import RandomForestClassifier
        if self.bRegressor :
            from sklearn.ensemble       import RandomForestRegressor
        self .metrics           = metrics
        self .distance_type:str = distance_type
        self .pdist             = lambda x : pdist(x,self.distance_type)
        self .RFC               = RandomForestClassifier
        if self.bRegressor :
            self.RFC            = RandomForestRegressor
        self .edge_importances_:np.array        = None
        self .edge_labels_:list[str]            = None
        self .computational_model_              = None # THE UNDERLYING CLASSIFIER MODEL
        self .model_funx        = tuple( ( lambda x:x/np.max(x) , lambda x: np.log(x+1) ) )
        self .func      = lambda x: np.sum(x)
        self .moms      = lambda x: np.array( [np.mean(x),np.std(x),scs.skew(x),-scs.kurtosis(x) , (np.mean(x)*np.std(x))**(1/3) ] )


class DjungleRegressor ( WildDjungle ) :
    pass

    def __init__ ( self , distance_type:str = 'euclidean' , bReturnDictionaries:bool=False ) :
        self.id_          :str  = ""
        self.description_ :str  = """A WRAPPER CLASS FOR A RANDOM FORERST CLASSSIFIER BUT FIRST EXPANDS MEASURES INTO DISTANCES"""
        self.model_label_ :str  = ""
        self.bRegressor:bool    = True
        self.bSimplePredict:bool = not bReturnDictionaries
        self.model_order_ :list = None
        self.array_order_ :list = None
        self.data_model_df      : pd.DataFrame  = None
        self.target_model_df    : pd.DataFrame  = None
        self.auxiliary_label_df : pd.DataFrame  = None
        self.bDataCompleted     : bool  = False
        self.descriptive_df     : pd.DataFrame  = None
        self.bg_label_          : str   = 'Background'
        self.fpr_ : np.array    = None
        self.tpr_ : np.array    = None
        self.auc_ : float       = None
        self.cvs_ : np.array    = None
        self.bDidFit_:bool      = False
        import scipy.stats as scs
        from scipy.spatial.distance     import pdist
        from sklearn                    import metrics
        from sklearn.ensemble           import RandomForestClassifier
        if self.bRegressor :
            from sklearn.ensemble       import RandomForestRegressor
        self .metrics           = metrics
        self .distance_type:str = distance_type
        self .pdist             = lambda x : pdist(x,self.distance_type)
        self .RFC               = RandomForestClassifier
        if self.bRegressor :
            self.RFC            = RandomForestRegressor
        self .edge_importances_:np.array        = None
        self .edge_labels_:list[str]            = None
        self .computational_model_              = None # THE UNDERLYING CLASSIFIER MODEL
        self .model_funx        = tuple( ( lambda x:x/np.max(x) , lambda x: np.log(x+1) ) )
        self .func      = lambda x: np.sum(x)
        self .moms      = lambda x: np.array( [np.mean(x),np.std(x),scs.skew(x),-scs.kurtosis(x) , (np.mean(x)*np.std(x))**(1/3) ] )



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
    # as of commit https://github.com/richardtjornhammar/RichTools/commit/99c79d94c2338252b1ef1067c0c061179b6edbd9 (YEAR:2016)
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
    # as of commit https://github.com/richardtjornhammar/RichTools/commit/99c79d94c2338252b1ef1067c0c061179b6edbd9 (YEAR:2016)
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

from impetuous.clustering import distance_matrix_to_absolute_coordinates
def HighDimensionalAlignment ( P:np.array , Q:np.array , bPdist:bool=True ) -> np.array :
    # HIGHER DIMENSIONAL VERSION OF
    # def KabschAlignment ( P , Q )
    #
    # https://en.wikipedia.org/wiki/Kabsch_algorithm
    # C++ VERSION: https://github.com/richardtjornhammar/RichTools/blob/master/src/richfit.cc
    #   IN VINCINITY OF LINE 524
    #
    # https://github.com/richardtjornhammar/RichTools/blob/master/src/richfit.cc
    # as of commit https://github.com/richardtjornhammar/RichTools/commit/99c79d94c2338252b1ef1067c0c061179b6edbd9 2016
    # SHAPE ALIGNMENT SEARCH FOR (shape_fit) SHAPE FIT
    #
    # THE DISTANCE GEMOETRY TO ABSOLUTE COORDINATES CAN BE FOUND HERE (2015)
    # https://github.com/richardtjornhammar/RichTools/commit/a6eef7c0712d1f87a20f319f951e09379a4171f0#diff-25a6634263c1b1f6fc4697a04e2b9904ea4b042a89af59dc93ec1f5d44848a26
    #
    # ALSO AN ALIGNMENT METHOD BUT NOT REDUCED TO ELLIPSOIDS WHERE THERE ARE SIGN AMBIGUITIES
    #
    # HERE P IS THE MODEL AND Q IS THE DATA
    # WE MOVE THE MODEL
    #
    if 'panda' in str(type(P)).lower() :
        P = P.values
    if 'panda' in str(type(Q)).lower() :
        Q = Q.values
    N , DIM  = np.shape( P )
    M , DIM  = np.shape( Q )
    P0 = P.copy()
    Q0 = Q.copy()
    #
    if DIM > N :
        print ( 'MALFORMED COORDINATE PROBLEM' )
        exit ( 1 )
    #
    if bPdist :
        from impetuous.clustering import absolute_coordinates_to_distance_matrix
        DP = absolute_coordinates_to_distance_matrix( P )
        DQ = absolute_coordinates_to_distance_matrix( Q )
    else :
        DP = np.array( [ np.sqrt(np.sum((p-q)**2)) for p in P for q in P ] ) .reshape( N,N )
        DQ = np.array( [ np.sqrt(np.sum((p-q)**2)) for p in Q for q in Q ] ) .reshape( M,M )
    #
    PX = distance_matrix_to_absolute_coordinates ( DP , n_dimensions = DIM ).T
    QX = distance_matrix_to_absolute_coordinates ( DQ , n_dimensions = DIM ).T
    #
    P = QX
    Q = Q
    #
    q0 , p0 , p0x = np.mean(Q,0) , np.mean(P,0), np.mean(PX,0)
    cQ , cP = Q - q0 , P - p0
    #
    H = np.dot(cP.T,cQ)
    I  = np.eye( DIM )
    #
    U, S, VT = np.linalg.svd( H, full_matrices = False )
    Ut = np.dot( VT.T,U.T )
    I[DIM-1,DIM-1] = 2*(np.linalg.det(Ut) > 0)-1
    ROT = np.dot( VT.T,np.dot(I,U.T) )
    #
    B = np.dot(ROT,PX.T).T + q0 - np.dot(ROT,p0x)
    #
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

from numpy import histogram as hist
def expect ( values , what:str='' ):
    nbins = int(np.ceil(len(values)*0.1))
    not_nans = []
    for v in values :
        if not 'nan' in str(v).lower():
            not_nans.append(v)
    if what == 'mean':
        return ( np.mean( not_nans ) )
    if what == 'median':
        return ( np.median(not_nans) )
    res = hist( not_nans , nbins )
    X   = ( res[1][1:]+res[1][:-1] )*0.5
    if what == 'represented':
        return (  X[np.argmax(res[0]) ] )
    return ( np.sum(X*res[0])/np.sum(res[0]) )

def impute_values ( df:pd.DataFrame , by:list=None , bVerbose:bool=True ) :
    if not by is None :
        sb = { s:i for i,s in zip(range(len(list(set(by)))),list(set(by))) }
    df.loc['gid'] = by
    impute_vals,imputed_df = None,None
    for analyte in df.index.values :
        if 'gid' in analyte :
            continue
        tmp = pd.DataFrame( df.loc[[analyte,'gid'],:].T.groupby('gid').apply(lambda x:pd.Series(expect(x.values),name=analyte) ) )
        tmp.columns = [analyte]
        w = []
        for v in  zip( df.loc[[analyte,'gid'],:].T.values ) :
            if 'nan' in str(v[0][0]).lower() :
                w.append( tmp.loc[v[0][1]].values[0] )
            else :
                w.append( v[0][0] )
        if imputed_df is None :
            imputed_df =  pd.DataFrame([w],index=[analyte] , columns=[df.columns] )
        else :
            imputed_df = pd.concat([ imputed_df,  pd.DataFrame([w],index=[analyte] , columns=[df.columns]) ])
        if impute_vals is None :
            impute_vals = tmp.T
        else :
            impute_vals = pd.concat([impute_vals,tmp.T])
        if bVerbose :
            print ( imputed_df,impute_vals , np.sum(imputed_df.values))
    impute_vals.index = df.index.values.tolist()[:-1]
    return ( imputed_df , impute_vals )


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
