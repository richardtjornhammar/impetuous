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
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import rankdata
from impetuous.convert import create_synonyms,flatten_dict
import itertools

def SubArraysOf(Array,Array_=None):
    if Array_ == None :
        Array_ = Array[:-1]
    if Array == []:
        if Array_ == []:
            return([])
        return( SubArraysOf(Array_,Array_[:-1]) )
    return([Array]+SubArraysOf(Array[1:],Array_))

def permuter( inputs , n ) :
    # permuter( inputs=['T2D','NGT','Female','Male'] , n=2 )
    return( [p[0] for p in zip(itertools.permutations(inputs,n))] )

def grouper ( inputs, n ):
    iters = [iter(inputs)] * n
    return zip(*iters)

from statsmodels.stats.multitest import multipletests
def adjust_p ( pvalue_list , method = 'fdr_bh' , alpha = 0.05,
               check_r_bh = False , is_sorted = False ,
               returnsorted = False
             ) :
    """  WRAPPER FOR MULTIPLE HYPOTHESIS TESTING
    pvalue_list = [0.00001,0.01,0.0002,0.00005,0.01,0.1,0.2,0.4,0.5,0.6,0.7,0.8,0.9,0.99,0.0114,0.15,0.23,0.20]
    """
    available_methods = set( [ 'bonferroni' , 'sidak',
           'holm-sidak' , 'holm' , 'simes-hochberg' ,
           'hommel' , 'fdr_bh' , 'fdr_by' , 'fdr_tsbh' ,
           'fdr_tsbky' ] )
    if method not in available_methods :
        print ( available_methods )
    r_equiv = { 'fdr_bh':'BH' }
    if check_r_bh and method in r_equiv :
        from rpy2.robjects.packages import importr
        from rpy2.robjects.vectors import FloatVector
        r_stats = importr('stats')
        p_adjust = r_stats.p_adjust ( FloatVector(pvalue_list), method = r_equiv[method] )
    else :
        p_adjust_results = multipletests ( pvalue_list, alpha=alpha, method=method, 
                       is_sorted = is_sorted , returnsorted = returnsorted )
        p_adjust = [ p_adj for p_adj in p_adjust_results[1] ]
    return ( p_adjust )

def qvalues ( p_values_in , pi0=None ) :
    p_s = p_values_in
    if pi0 is None :
        pi0 = 1
    qs_ = []
    m = float(len(p_s)) ; itype = str( type( p_s[0] ) ) ; added_info = False
    if 'list' in itype or 'tuple' in itype :
        added_info = True
        ps = [ p[0] for p in p_s ]
    else :
        ps = p_s
    frp_ = rankdata( ps,method='ordinal' )/m
    ifrp_ = [ ( (p<=f)*f + p*(p>f) ) for p,f in zip(ps,frp_) ]
    for ip in range(len(ps)) :
        p_ = ps[ ip ] ; f_ = frp_[ip]
        q_ = p_ / ifrp_[ip]
        qs_.append( (q_,p_) )
    if added_info :
        q   = [ tuple( [qvs[0]]+list(pinf) ) for ( qvs,pinf ) in zip(qs_,p_s) ]
        qs_ = q
    return qs_

from scipy import stats
from statsmodels.stats.anova import anova_lm as anova
import statsmodels.api as sm
import patsy
def anova_test ( formula, group_expression_df, journal_df, test_type = 'random' ) :
    type_d = { 'paired':1 , 'random':2 , 'fixed':1 }
    formula = formula.replace(' ','')
    tmp_df = pd.concat([ journal_df, group_expression_df ])
    gname = tmp_df.index.tolist()[-1]
    formula_l = formula.split('~')
    rename = { gname:formula_l[0] }
    tmp_df.rename( index=rename, inplace=True )
    tdf = tmp_df.T.iloc[ :,[ col in formula for col in tmp_df.T.columns] ].apply( pd.to_numeric )
    y, X = patsy.dmatrices( formula, tdf, return_type='dataframe')
    model = sm.OLS(endog=y,exog=X).fit()
    model .model.data.design_info = X.design_info
    table = sm.stats.anova_lm(model,typ=type_d[test_type])
    return table.iloc[ [(idx in formula) for idx in table.index],-1]

from scipy.stats import ttest_rel,ttest_ind,mannwhitneyu
def t_test ( df , endogen = 'expression' , group = 'disease' ,
             pair_values = ('Sick','Healthy') , test_type = 'independent',
             equal_var = False, alternative = 'greater' ) :
    group1 = df[df[group] == pair_values[0]][endogen].astype(float)
    group2 = df[df[group] == pair_values[1]][endogen].astype(float)
    if test_type == 'independent':
        pv = ttest_ind( group1, group2 , equal_var = equal_var )
    if test_type == 'related':
        pv = ttest_rel( group1, group2 )
    try :
        p_normality = mannwhitneyu( group1, group2, alternative=alternative )[1]
    except ValueError as err:
        print(err.args)
        p_normality = 1.0    
    pvalue = pv[1] ; statistic=pv[0]
    return ( pvalue , p_normality, statistic )

def parse_test ( statistical_formula, group_expression_df , journal_df , test_type = 'random' ) :
    if 'glm' in statistical_formula.lower():
        print ( 'NOT IMPLEMENTED YET' )
    #
    # THE FALLBACK IS A TYPE2 ANOVA
    if 'ttest' in statistical_formula.lower() :
        result = None
        # WE CONDUCT SEPARATE TESTS FOR ALL THE UNIQUE PAIR LABELS PRESENT
        check = [ idx for idx in journal_df.index if idx in statistical_formula ]
        df = pd.concat( [journal_df,group_expression_df],axis=0 ).T
        for c in check :
            if test_type in set([ 'related' , 'fixed' , 'paired' ]):
                test_type = 'related'
            else :
                test_type = 'independent'

            for pair in permuter( list(set(journal_df.loc[c].values)),2) :
                result_ = t_test( df, endogen = df.columns.values[-1], group = c,
                                  pair_values = pair, test_type = test_type, equal_var = False )
                hdr = ' '.join( [c,' '.join([str(p) for p in pair])] )
                tdf = pd.Series( result_, index = [ hdr, hdr+' mwu', hdr+' stat,s' ] )
                if result is None :
                    result = tdf
                else :
                    result = pd.concat([result,tdf])
                result.name = 'PR>t'
    else :
        result = anova_test( statistical_formula, group_expression_df , journal_df , test_type=test_type )
    return ( result )

def prune_journal ( journal_df , remove_units_on = '_' ) :
    journal_df = journal_df.loc[ [ 'label' in idx.lower() or '[' in idx for idx in journal_df.index.values] , : ].copy()
    bSel = [ ('label' in idx.lower() ) for idx in journal_df.index.values]
    bool_dict = { False:0 , True:1 , 'False':0 , 'True':1 }
    str_journal = journal_df.iloc[ bSel ]
    journal_df = journal_df.replace({'ND':np.nan})
    nmr_journal = journal_df.iloc[ [ not b for b in bSel ] ].replace(bool_dict).apply( pd.to_numeric )
    if not remove_units_on is None :
        nmr_journal.index = [ idx.split(remove_units_on)[0] for idx in nmr_journal.index ]
    journal_df = pd.concat( [nmr_journal,str_journal] )
    return( journal_df )

def merge_significance ( significance_df , distance_type='euclidean' ) :
    # TAKES P VALUES OR Q VALUES
    # TRANSFORMS INTO A MERGED P OR Q VALUE VIA
    # THE DISTANCE SCORE
    # THE DATA DOMAIN SIGNIFICANCE IS ALONG COLUMNS AND
    # GROUPS ALONG INDICES
    # EX: pd.DataFrame( np.random.rand(20).reshape(5,4) , columns=['bio','cars','oil','money']).apply( lambda x: -1.*np.log10(x) ).T.apply( lambda x: np.sqrt(np.sum(x**2)) )
    #
    distance = lambda x : np.sqrt(np.sum(x**2))
    if distance_type == 'euclidean' : # ESTIMATE
        distance = lambda x : np.sqrt(np.sum(x**2))
    if distance_type == 'extreme' :   # ANTI-CONSERVATIVE ESTIMATE
        distance = lambda x : np.max(x)
    get_pvalue = lambda x : 10**(-x)
    return ( significance_df.apply( lambda x: -1.*np.log10(x) ).T.apply(distance).apply(get_pvalue) )

def group_significance( subset , all_analytes_df = None ,
                        tolerance = 0.05 , significance_name = 'pVal' ,
                        AllAnalytes = None , SigAnalytes = None,
                        alternative = 'greater' ) :
    # FISHER ODDS RATIO CHECK
    # CHECK FOR ALTERNATIVE :
    #   'greater'   ( ENRICHMENT IN GROUP )
    #   'two-sided' ( DIFFERENTIAL GROUP EXPERSSION )
    #   'less'      ( DEPLETION IN GROUP )
    if AllAnalytes is None :
        if all_analytes_df is None :
            AllAnalytes = set(all_analytes_df.index.values)
    if SigAnalytes is None :
        if all_analytes_df is None :
            SigAnalytes = set( all_analytes_df.iloc[(all_analytes_df<tolerance).loc[:,significance_name]].index.values )
    Analytes       = set(subset.index.values)
    notAnalytes    = AllAnalytes - Analytes
    notSigAnalytes = AllAnalytes - SigAnalytes
    AB  = len(Analytes&SigAnalytes)    ; nAB  = len(notAnalytes&SigAnalytes)
    AnB = len(Analytes&notSigAnalytes) ; nAnB = len(notAnalytes&notSigAnalytes)
    oddsratio , pval = stats.fisher_exact([[AB, nAB], [AnB, nAnB]], alternative=alternative)
    return ( pval , oddsratio )


def quantify_groups_by_analyte_pvalues( analyte_df, grouping_file, delimiter='\t',
                                 tolerance = 0.05 , p_label = 'C(Status),p' ,
                                 group_prefix = '' ) :
    AllAnalytes = set( analyte_df.index.values ) ; nidx = len( AllAnalytes )
    SigAnalytes = set( analyte_df.iloc[ (analyte_df.loc[:,p_label].values < tolerance), : ].index.values )
    if len(AllAnalytes) == len(SigAnalytes) :
        print ( 'THIS STATISTICAL TEST WILL BE NONSENSE' )
    eval_df = None
    with open( grouping_file ) as input :
        for line in input :
            vline = line.replace('\n','').split(delimiter)
            gid, gdesc, analytes_ = vline[0], vline[1], vline[2:]
            try :
                group = analyte_df.loc[[a for a in analytes_ if a in AllAnalytes] ].dropna( axis=0, how='any', thresh=analyte_df.shape[1]/2 ).drop_duplicates()
            except KeyError as e :
                continue
            L_ = len( group ) ; str_analytes=','.join(group.index.values)
            if L_ > 0 :
                pv,odds = group_significance( group , AllAnalytes=AllAnalytes, SigAnalytes=SigAnalytes )
                rdf = pd.DataFrame( [[pv]], columns = [ group_prefix + 'Fisher_'+p_label ], index=[ gid ] )
                rdf.columns = [ col+',p' if ',p' not in col else col for col in rdf.columns ]
                rdf[ 'description' ] = gdesc+',' + str(L_) ; rdf['analytes'] = str_analytes 
                rdf[ group_prefix + 'NGroupAnalytes' ] = L_
                rdf[ group_prefix + 'FracGroupFill' ]  = L_ / float(len(analytes_))
                ndf = rdf
                if eval_df is None :
                    eval_df = ndf
                else :
                    eval_df = pd.concat([eval_df,ndf])
    edf = eval_df.T
    for col in eval_df.columns :
        if ',p' in col :
            q = [q_[0] for q_ in qvalues(eval_df.loc[:,col].values)]; l=col.split(',')[0]+',q'
            edf.loc[l] = q
    return ( edf.T )

dimred = PCA()
def quantify_groups ( analyte_df , journal_df , formula , grouping_file , synonyms = None ,
                      delimiter = '\t' , test_type = 'random' ,
                      split_id = None , skip_line_char = '#' ) :
    statistical_formula = formula
    if not split_id is None :
        nidx = [ idx.split(split_id)[-1].replace(' ','') for idx in analyte_df.index.values ]
        analyte_df.index = nidx
    sidx = set( analyte_df.index.values ) ; nidx=len(sidx)
    eval_df = None
    with open ( grouping_file ) as input:
        for line in input:
            if line[0] == skip_line_char :
                continue
            vline = line.replace('\n','').split(delimiter)
            gid,gdesc,analytes_ = vline[0],vline[1],vline[2:]
            if not synonyms is None :
                [ analytes_.append(synonyms[a]) for a in analytes_ if a in synonyms ]
            try :
                group = analyte_df.loc[[a for a in analytes_ if a in sidx] ].dropna( axis=0, how='any', thresh=analyte_df.shape[1]/2 ).drop_duplicates()
            except KeyError as e :
                continue
            L_ = len( group ); str_analytes=','.join(group.index.values)
            if L_>0 :
                dimred.fit(group.values)
                group_expression_df = pd.DataFrame([dimred.components_[0]],columns=analyte_df.columns.values,index=[gid])
                rdf = pd.DataFrame( parse_test( statistical_formula, group_expression_df , journal_df , test_type=test_type )).T
                rdf .columns = [ col+',p' if (not ',s' in col) else col+',s' for col in rdf.columns ]
                rdf['description'] = gdesc+','+str(L_)
                rdf['analytes'] = str_analytes
                rdf.index = [ gid ] ; ndf = pd.concat([rdf.T,group_expression_df.T]).T
                if eval_df is None :
                    eval_df = ndf
                else :
                    eval_df = pd.concat([eval_df,ndf])
    edf = eval_df.T
    for col in eval_df.columns :
        if ',p' in col :
            q = [q_[0] for q_ in qvalues(eval_df.loc[:,col].values)]; l=col.split(',')[0]+',q'
            edf.loc[l] = q
    return ( edf.T )

def quantify_analytes( analyte_df, journal_df, formula,
                       delimiter = '\t', test_type = 'random',
                       verbose = True, only_include = None ) :
    statistical_formula = formula
    sidx = set(analyte_df.index.values) ; nidx=len(sidx)
    eval_df = None ; N_ = len(analyte_df)
    for iline in range ( len( analyte_df ) ) :
        group = analyte_df.iloc[ [iline],: ]
        if 'str' in str(type(only_include)) :
            if not only_include in group.index :
                continue
        L_ = len ( group ) ; str_analytes = '.'.join( group.index.values )
        if L_>0 :
            gid = group.index.values[0].split('.')[-1].replace(' ','') ; gdesc = group.index.values[0].split('.')[0]
            group_expression_df = pd.DataFrame([group.values[0]], columns=analyte_df.columns.values, index=[gid] )
            rdf = pd.DataFrame(parse_test( statistical_formula, group_expression_df, journal_df, test_type=test_type )).T
            rdf .columns = [ col+',p' if (not ',s' in col) else col+',s' for col in rdf.columns ]
            rdf['description'] = gdesc+','+str(L_)
            rdf['analytes'] = str_analytes
            rdf .index = [ gid ] ; ndf = pd.concat([rdf.T,group_expression_df.T]).T
            if eval_df is None :
                eval_df = ndf
            else :
                eval_df = pd.concat([eval_df,ndf])
        if verbose :
            print ( 'Done:', str(np.floor(float(iline)/N_*1000.)/10.)+'%'  , end="\r")
    edf = eval_df.T
    for col in eval_df.columns :
        if ',p' in col :
            q = [q_[0] for q_ in qvalues(eval_df.loc[:,col].values)]; l=col.split(',')[0]+',q'
            edf.loc[l] = q
    return ( edf.T )

def group_counts( analyte_df, grouping_file, delimiter = '\t',
                  tolerance = 0.05 , p_label = 'C(Status),p' ,
                  group_prefix = ''
                ) :
    AllAnalytes = list( analyte_df.index.values )
    AllAnalytes = set( AllAnalytes ) ; nidx = len( AllAnalytes )
    SigAnalytes = set( analyte_df.iloc[ (analyte_df.loc[:,p_label].values < tolerance), : ].index.values )
    eval_df = None
    with open( grouping_file ) as input :
        for line in input :
            vline = line.replace('\n','').split(delimiter)
            gid, gdesc, analytes_ = vline[0], vline[1], vline[2:]
            useSigAnalytes = [ a for a in analytes_ if a in SigAnalytes ]
            try :
                group = analyte_df.loc[ useSigAnalytes ].dropna( axis=0, how='any', thresh=analyte_df.shape[1]/2 ).drop_duplicates()
            except KeyError as e :
                continue
            L_ = len( group ) ; str_analytes=','.join(group.index.values)
            if L_ > 0 :
                rdf = pd.DataFrame( [[L_,L_/float(len(analytes_))]], columns = [ 
                          group_prefix + 'NsigFrom' + p_label , 
                          group_prefix + 'FsigFrom' + p_label ] , index = [ gid ] )
                ndf = rdf
                if eval_df is None :
                    eval_df = ndf
                else :
                    eval_df = pd.concat ( [eval_df,ndf] )
    return ( eval_df.sort_values( group_prefix + 'FsigFrom' + p_label ) )

def retrieve_genes_of( group_name, grouping_file, delimiter='\t', identifier='ENSG', skip_line_char='#' ):
    all_analytes = []
    with open ( grouping_file ) as input:
        for line_ in input :
            if skip_line_char == line_[0] :
                continue
            if group_name is None :
                line  = line_.replace( '\n','' )
                dline = line.split( delimiter )
                if identifier is None : 
                    [ all_analytes.append(d) for d in dline ]
                else :
                    [ all_analytes.append(d) for d in dline if identifier in d ]
            else:
                if group_name in line_:
                    line=line_.replace('\n','')
                    dline = line.split(delimiter)
                    if identifier is None : 
                        return ( [ d for d in dline ] )
                    else:
                        return ( [ d for d in dline if identifier in d ] )
    return ( list(set(all_analytes)) )

import math
def differential_analytes ( analyte_df , cols = [['a'],['b']] ):
    adf = analyte_df.loc[ :,cols[0] ].copy().apply(pd.to_numeric)
    bdf = analyte_df.loc[ :,cols[1] ].copy().apply(pd.to_numeric)
    regd_l = ( adf.values - bdf.values )
    regd_r = -regd_l
    ddf = pd.DataFrame( np.array( [ regd_l.reshape(-1) , regd_r.reshape(-1) , regd_l.reshape(-1)**2 ] ).T ,
                        columns=['DiffL','DiffR','Dist'] , index=adf.index )
    for col in adf.columns :
        ddf.loc[:,col] = adf.loc[:,col]
    for col in bdf.columns :
        ddf.loc[:,col] = bdf.loc[:,col]
    ddf = ddf.sort_values('Dist', ascending = False )
    return ( ddf )

def add_kendalltau( analyte_results_df, journal_df, what='M') :
    if what in set(journal_df.index.values) :
        # ADD IN CONCOORDINANCE WITH KENDALL TAU
        from scipy.stats import kendalltau
        K = []
        patients = [ c for c in analyte_results_df.columns if '_' in c ]
        for idx in analyte_results_df.index :
            y = journal_df.loc[what,patients].values
            x = analyte_results_df.loc[[idx],patients].values[0] # IF DUPLICATE GET FIRST
            k = kendalltau( x,y )
            K .append( k )
        analyte_results_df['KendallTau'] = K
    return ( analyte_results_df )

if __name__ == '__main__' :

    test_type = 'random'

    path_ = './'
    analyte_file  = path_ + 'fine.txt'
    journal_file  = path_ + 'coarse.txt'
    grouping_file = path_ + 'groups.gmt'

    analyte_df = pd.read_csv(analyte_file,'\t' , index_col=0 )
    journal_df = prune_journal( pd.read_csv(journal_file,'\t', index_col=0 ) )

    print ( quantify_groups( analyte_df, journal_df, 'Group ~ Var + C(Cat) ', grouping_file ) )
