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

def qvalues ( p_values_in , pi0=None ) :
    p_s = p_values_in
    if pi0 is None:
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
def anova_test( formula, group_expression_df , journal_df ):
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
    table = sm.stats.anova_lm(model,typ=2)
    return table.iloc[ [(idx in formula) for idx in table.index],-1]

def prune_journal( journal_df,remove_units_on='_' ):
    bSel = [ ('label' in idx.lower() ) for idx in journal_df.index.values] 
    bool_dict = { False:0 , True:1 , 'False':0 , 'True':1 }
    str_journal = journal_df.iloc[ bSel ]
    nmr_journal = journal_df.iloc[ [ not b for b in bSel ] ].replace(bool_dict).apply( pd.to_numeric )
    if not remove_units_on is None:
        nmr_journal.index = [ idx.split(remove_units_on)[0] for idx in nmr_journal.index ]
    journal_df = pd.concat( [nmr_journal,str_journal] )
    return( journal_df )

dimred = PCA()

def quantify_groups( analyte_df,journal_df,formula,grouping_file,delimiter='\t' ) :
    statistical_formula=formula
    sidx = set(analyte_df.index.values); nidx=len(sidx)
    eval_df = None
    with open(grouping_file) as input:
        for line in input:
            vline = line.replace('\n','').split(delimiter)
            gid,gdesc,analytes_ = vline[0],vline[1],vline[2:]
            group = analyte_df.loc[[a for a in analytes_ if a in sidx] ].dropna()
            L_ = len( group ); str_analytes=','.join(group.index.values)
            if L_>0 :
                dimred.fit(group.values)
                group_expression_df = pd.DataFrame([dimred.components_[0]],columns=analyte_df.columns.values,index=[gid])
                rdf = pd.DataFrame(anova_test( statistical_formula, group_expression_df , journal_df )).T
                rdf.columns = [ col+',p' for col in rdf.columns ]
                rdf['description'] = gdesc+','+str(L_)
                rdf['analytes'] = str_analytes
                rdf.index = [ gid ]; ndf = pd.concat([rdf.T,group_expression_df.T]).T
                if eval_df is None:
                    eval_df = ndf
                else:
                    eval_df = pd.concat([eval_df,ndf])
    edf = eval_df.T
    for col in eval_df.columns :
        if ',p' in col :
            q = [q_[0] for q_ in qvalues(eval_df.loc[:,col].values)]; l=col.split(',')[0]+',q'
            edf.loc[l] = q
    return( edf.T )

if __name__ == '__main__' :
    path = './'
    analyte_file = path_ + 'fine.txt'
    journal_file = path_ + 'coarse.txt'
    grouping_file = path_ + 'groups.gmt'

    analyte_df = pd.read_csv(analyte_file,'\t' , index_col=0 )
    journal_df = prune_journal( pd.read_csv(journal_file,'\t', index_col=0 ) )

    print ( quantify_groups( analyte_df, journal_df, 'Group ~ Var + C(Cat) ', grouping_file ) )

