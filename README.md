# A Statistical Learning library for Humans
This toolkit currently offers enrichment analysis, hierarchical enrichment analysis, novel PLS regression, shape alignment, connectivity clustering, clustering and hierarchical clustering as well as factor analysis methods.

The fine grained data can be studied via a statistical tests that relates it to observables in a coarse grained journal file. The final p values can then be rank corrected by employing a novel q-value method.

[![Downloads](https://pepy.tech/badge/impetuous-gfa)](https://pepy.tech/project/impetuous-gfa)
[![Downloads](https://pepy.tech/badge/impetuous-gfa/month)](https://pepy.tech/project/impetuous-gfa)
[![Downloads](https://pepy.tech/badge/impetuous-gfa/week)](https://pepy.tech/project/impetuous-gfa)

Visit the active code via :
https://github.com/richardtjornhammar/impetuous

Visit the published code : 
https://doi.org/10.5281/zenodo.2594690

Cite using :
DOI: 10.5281/zenodo.2594690

# Pip installation with :
```
pip install impetuous-gfa
```

# Version controlled installation of the Impetuous library

The Impetuous library

In order to run these code snippets we recommend that you download the nix package manager. Nix package manager links from Oktober 2020:

https://nixos.org/download.html

```
$ curl -L https://nixos.org/nix/install | sh
```

If you cannot install it using your Wintendo then please consider installing Windows Subsystem for Linux first:

```
https://docs.microsoft.com/en-us/windows/wsl/install-win10
```

In order to run the code in this notebook you must enter a sensible working environment. Don't worry! We have created one for you. It's version controlled against python3.7 and you can get the file here:

https://github.com/richardtjornhammar/rixcfgs/blob/master/code/environments/impetuous-shell.nix

Since you have installed Nix as well as WSL, or use a Linux (NixOS) or bsd like system, you should be able to execute the following command in a termnial:

```
$ nix-shell impetuous-shell.nix
```

Now you should be able to start your jupyter notebook locally:

```
$ jupyter-notebook impetuous_finance.ipynb
```

and that's it.

# Usage example 1: Elaborate informatics

code: https://gitlab.com/stochasticdynamics/eplsmta-experiments
docs: https://arxiv.org/pdf/2001.06544.pdf

# Usage example 2: Simple regression code

Now while in a good environment: In your Jupyter notebook or just in a dedicated file.py you can write the following:

```
import pandas as pd
import numpy as np

import impetuous.quantification as impq

analyte_df = pd.read_csv( 'analytes.csv' , '\t' , index_col=0 )
journal_df = pd.read_csv( 'journal.csv'  , '\t' , index_col=0 )

formula = 'S ~ C(industry) : C(block) + C(industry) + C(block)'

res_dfs 	= impq.run_rpls_regression ( analyte_df , journal_df , formula , owner_by = 'angle' )
results_lookup	= impq.assign_quality_measures( journal_df , res_dfs , formula )

print ( results_lookup )
print ( res_dfs )
```

# Usage example 3: Novel NLP sequence alignment

Finding a word in a text is a simple and trivial problem in computer science. However matching a sequence of characters to a larger text segment is not. In this example you will be shown how to employ the impetuous text fitting procedure. The strength of the fit is conveyed via the returned score, higher being a stronger match between the two texts. This becomes costly for large texts and we thus break the text into segments and words. If there is a strong word to word match then the entire segment score is calculated. The off and main diagonal power terms refer to how to evaluate a string shift. Fortinbras and Faortinbraaks are probably the same word eventhough the latter has two character shifts in it. In this example both "requests" and "BeautifulSoup" are employed to parse internet text.

```
import numpy as np
import pandas as pd

import impetuous.fit as impf    # THE IMPETUOUS FIT MODULE
                                # CONTAINS SCORE ALIGNMENT ROUTINE

import requests                 # FOR MAKING URL REQUESTS
from bs4 import BeautifulSoup   # FOR PARSING URL REQUEST CONTENT

if __name__ == '__main__' :

    print ( 'DOING TEXT SCORING VIA MY SEQUENCE ALIGNMENT ALGORITHM' )
    url_       = 'http://shakespeare.mit.edu/hamlet/full.html'

    response   = requests.get( url_ )
    bs_content = BeautifulSoup ( response.content , features="html.parser")

    name = 'fortinbras'
    score_co = 500
    S , S2 , N = 0 , 0 , 0
    for btext in bs_content.find_all('blockquote'):

        theTextSection = btext.get_text()
        theText        = theTextSection.split('\n')

        for segment in theText:
            pieces = segment.split(' ')
            if len(pieces)>1 :
                for piece in pieces :
                    if len(piece)>1 :
                        score = impf.score_alignment( [ name , piece ],
                                    main_diagonal_power = 3.5, shift_allowance=2,
                                    off_diagonal_power = [1.5,0.5] )
                        S    += score
                        S2   += score*score
                        N    += 1
                        if score > score_co :
                            print ( "" )
                            print ( score,name,piece )
                            print ( theTextSection )
                            print ( impf.score_alignment( [ name , theTextSection ],
                                        main_diagonal_power = 3.5, shift_allowance=2,
                                        off_diagonal_power = [1.5,0.5] ) )
                            print ( "" )

    print ( S/N )
    print ( S2/N-S*S/N/N )
```

# Usage example 4 : Diabetes analysis

Here we show how to use a novel multifactor method on a diabetes data set to deduce important transcripts with respect to being diabetic. The data was obtained from the [Broad Insitute](http://www.gsea-msigdb.org/gsea/datasets.jsp) and contains gene expressions from a microarray hgu133a platform. We choose to employ the `Diabetes_collapsed_symbols.gct` file since it has already been collapsed down to useful transcripts. We have entered an `impetuous-gfa` ( version >= `0.50.0` ) environment and set up the a `diabetes.py` file with the following code content:

```
import pandas as pd
import numpy as np

if __name__ == '__main__' :
    analyte_df = pd.read_csv('../data/Diabetes_collapsed_symbols.gct','\t', index_col=0, header=2).iloc[:,1:]
```

In order to illustrate the use of low value supression we use the reducer module.  A `tanh` based soft max function is employed by the confred function to supress values lower than the median of the entire sample series for each sample.
```
    from impetuous.reducer import get_procentile,confred
    for i_ in range(len(analyte_df.columns.values)):
        vals   = analyte_df.iloc[:,i_].values
        eta    = get_procentile( vals,50 )
        varpi  = get_procentile( vals,66 ) - get_procentile( vals,33 )
        analyte_df.iloc[:,i_] = confred(vals,eta,varpi)

    print ( analyte_df )
```

The data now contain samples along the columns and gene transcript symbols along the rows where the original values have been quenched with low value supression. The table have the following appearance

|NAME       |NGT_mm12_10591 | ... | DM2_mm81_10199 |
|:---       |           ---:|:---:|            ---:|
|215538_at  |    16.826041 | ... | 31.764484       |
|...        |              |     |                 |
|LDLR       |   19.261185  | ... | 30.004612       |

We proceed to write a journal data frame by adding the following lines to our code
```
    journal_df = pd.DataFrame([ v.split('_')[0] for v in analyte_df.columns] , columns=['Status'] , index = analyte_df.columns.values ).T
    print ( journal_df )
```
which will produce the following journal table :

|      |NGT_mm12_10591 | ... | DM2_mm81_10199 |
|:---    |         ---:|:---:|            ---:|
|Status  |         NGT | ... | DM2            |

Now we check if there are aggregation tendencies among these two groups prior to the multifactor analysis. We could use the hierarchical clustering algorithm, but refrain from it and instead use the `associations` method together with the `connectivity` clustering algorithm. The `associations` can be thought of as a type of ranked correlations similar to spearman correlations. If two samples are strongly associated with each other they will be close to `1` (or `-1` if they are anti associated). Since they are all humans, with many transcript features, the values will be close to `1`. After recasting the `associations` into distances we can determine if two samples are connected at a given distance by using the `connectivity` routine. All connected points are then grouped into technical clusters, or batches, and added to the journal.
```
    from impetuous.quantification import associations
    ranked_similarity_df = associations ( analyte_df .T )
    sample_distances = ( 1 - ranked_similarity_df ) * 2.

    from impetuous.clustering import connectivity
    cluster_ids = [ 'B'+str(c[0]) for c in connectivity( sample_distances.values , 5.0E-2 )[1] ]
    print ( cluster_ids )

    journal_df .loc['Batches'] = cluster_ids
```
which will produce a cluster list containing `13` batches with members whom are `Normal Glucose Tolerant` or have `Diabetes Mellitus 2`. We write down the formula for deducing which genes are best at recreating the diabetic state and batch identities by writing:
```
    formula = 'f~C(Status)+C(Batches)'
```
The multifactor method calculates how to produce an encoded version of the journal data frame given an analyte data set. It does this by forming the psuedo inverse matrix that best describes the inverse of the analyte frame and then calculates the dot product of the inverse with the encoded journal data frame. This yields the coefficient frame needed to solve for the numerical encoding frame. The method has many nice statistical properties that we will not discuss further here. The first thing that the multifactor method does is to create the encoded data frame. The encoded data frame for this problem can be obtained with the following code snippet
```
    encoded_df = create_encoding_data_frame ( journal_df , formula ).T
    print ( encoded_df )
```
and it will look something like this

|      |NGT_mm12_10591 | ... | DM2_mm81_10199 |
|:---  |           ---:|:---:|            ---:|
|B10   |         0.0   | ... | 0.0            |
|B5    |         0.0   | ... | 0.0            |
|B12   |         0.0   | ... | 1.0            |
|B2    |         0.0   | ... | 0.0            |
|B11   |         1.0   | ... | 0.0            |
|B8    |         0.0   | ... | 0.0            |
|B1    |         0.0   | ... | 0.0            |
|B7    |         0.0   | ... | 0.0            |
|B4    |         0.0   | ... | 0.0            |
|B0    |         0.0   | ... | 0.0            |
|B6    |         0.0   | ... | 0.0            |
|B9    |         0.0   | ... | 0.0            |
|B3    |         0.0   | ... | 0.0            |
|NGT   |         1.0   | ... | 0.0            |
|DM2   |         0.0   | ... | 1.0            |

This encoded dataframe can be used to calculate statistical parameters or solve other linear equations. Take the fast calculation of the mean gene expressions across all groups as an example
```
    print ( pd .DataFrame ( np.dot( encoded_df,analyte_df.T ) ,
                          columns = analyte_df .index ,
                          index   = encoded_df .index ) .apply ( lambda x:x/np.sum(encoded_df,1) ) )
```
which will immediately calculate the mean values of all transcripts across all different groups.

The `multifactor_evaluation` calculates the coefficients that best recreates the encoded journal by employing the psudo inverse of the analyte frame utlizing Singular Value Decomposition. The beta coefficients are then evaluated using a normal distribution assumption to obtain `p values` and rank corrected `q values` are also returned. The full function can be called with the following code
```
    from impetuous.quantification import multifactor_evaluation
    multifactor_results = multifactor_evaluation ( analyte_df , journal_df , formula )

    print ( multifactor_results.sort_values('DM2,q').iloc[:25,:].index.values  )
```
which tells us that the genes
```
['MYH2' 'RPL39' 'HBG1 /// HBG2' 'DDN' 'UBC' 'RPS18' 'ACTC' 'HBA2' 'GAPD'
 'ANKRD2' 'NEB' 'MYL2' 'MT1H' 'KPNA4' 'CA3' 'RPLP2' 'MRLC2 /// MRCL3'
 '211074_at' 'SLC25A16' 'KBTBD10' 'HSPA2' 'LDHB' 'COX7B' 'COX7A1' 'APOD']
```
have something to do with the altered metabolism in Type 2 Diabetics. We could now proceed to use the hierarchical enrichment routine to understand what that something is, but first we save the data
```
    multifactor_results.to_csv('multifactor_dm2.csv','\t')
```

# Example 5: Understanding what it means

If you have a well curated `.gmt` file that contains analyte ids as unique sets that belong to different groups then you can check whether or not a specific group seems significant with respect to all of the significant and insignificant analytes that you just calculated. One can derive such a hierarchy or rely on already curated information. Since we are dealing with genes and biologist generally have strong opinions about these things we go to a directed acyclic knowledge graph called [Reactome](https://reactome.org/PathwayBrowser/) and translate that information into a set of [files](https://zenodo.org/record/3608712) that we can use to build our own knowledge hierarchy. After downloading that `.zip` file (and unzipping) you will be able to execute the following code
```
import pandas as pd
import numpy as np

if __name__=='__main__':
    import impetuous.pathways as impw
    impw.description()
```
which will blurt out code you can use as inspiration to generate the Reactome knowledge hierarchy. So now we do that
```
    paths = impw.Reactome( './Ensembl2Reactome_All_Levels_v71.txt' )
```
but we also need to translate the gene ids into the correct format so we employ [BioMart](http://www.ensembl.org/biomart/martview). To obtain the conversion text file we select `Human genes GRCh38.p13` and choose attributes `Gene stable ID`, `Gene name` and `Gene Synonym` and save the file as `biomart.txt`.

```
    biomart_dictionary = {}
    with open('biomart.txt','r') as input:
        for line in input :
            lsp = line.split('\n')[0].split('\t')
            biomart_dictionary[lsp[0]] = [ n for n in lsp[1:] if len(n)>0 ]
    paths.add_pathway_synonyms( synonym_dict=biomart_dictionary )

    paths .make_gmt_pathway_file( './reactome_v71.gmt' )
```
Now we are almost ready to conduct the hierarchical pathway enrichment, to see what cellular processes are significant with respect to our gene discoveries, but we still need to build the Directed Acyclic Graph (DAG) from the parent child file and the pathway definitions.
```
    import impetuous.hierarchical as imph
    dag_df , tree = imph.create_dag_representation_df ( pathway_file = './reactome_v71.gmt',
                                                        pcfile = './NewestReactomeNodeRelations.txt' )
```
We will use it in the `HierarchicalEnrichment` routine later in order not to double count genes that have already contributed at lower levels of the hierarchy. Now where did we store those gene results...
```
    quantified_analyte_df = pd.read_csv('multifactor_dm2.csv','\t',index_col=0)
    a_very_significant_cutoff = 1E-10

    enrichment_results = imph.HierarchicalEnrichment ( quantified_analyte_df , dag_df , 
                                  ancestors_id_label = 'DAG,ancestors' , dag_level_label = 'DAG,level' ,
                                  threshold = a_very_significant_cutoff ,
                                  p_label = 'DM2,q' )
```
lets see what came out on top!
```
    print( enrichment_results.sort_values('Hierarchical,p').loc[:,['description','Hierarchical,p']].iloc[0,:] )
```
which will report that

|description | Striated Muscle Contraction  |
|:---       |           ---:|
|Hierarchical,p  |   6.55459e-05 |
|Name: |  R-HSA-390522 |

is affected or perhaps needs to be compensated for... now perhaps you thought this exercise was a tad tedious? Well you are correct. It is and you could just as well have copied the gene transcripts into [String-db](https://string-db.org/cgi/input?sessionId=beIptQQxF85j&input_page_active_form=multiple_identifiers) and gotten similar results out. But, then you wouldn't have gotten to use the hierarchical enrichment method I invented!

# Example 6: Absolute and relative coordinates

In this example, we will use the SVD based distance geometry method to go between absolute coordinates, relative coordinate distances and back to ordered absolute coordinates. Absolute coordinates are float values describing the position of something in space. If you have several of these then the same information can be conveyed via the pairwise distance graph. Going from absolute coordinates to pairwise distances is simple and only requires you to calculate all the pairwise distances between your absolute coordinates. Going back to mutually orthogonal ordered coordinates from the pariwise distances is trickier, but a solved problem. The [distance geometry](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.37.8051) can be obtained with SVD and it is implemented in the `impetuous.clustering` module under the name `distance_matrix_to_absolute_coordinates`. We start by defining coordinates afterwhich we can calculate the pair distance matrix and transforming it back by using the code below

```
import pandas as pd
import numpy as np

coordinates = np.array([[-23.7100 ,  24.1000 ,  85.4400],
  [-22.5600 ,  23.7600 ,  85.6500],
  [-21.5500 ,  24.6200 ,  85.3800],
  [-22.2600 ,  22.4200 ,  86.1900],
  [-23.2900 ,  21.5300 ,  86.4800],
  [-20.9300 ,  22.0300 ,  86.4300],
  [-20.7100 ,  20.7600 ,  86.9400],
  [-21.7900 ,  19.9300 ,  87.1900],
  [-23.0300 ,  20.3300 ,  86.9600],
  [-24.1300 ,  19.4200 ,  87.2500],
  [-23.7400 ,  18.0500 ,  87.0000],
  [-24.4900 ,  19.4600 ,  88.7500],
  [-23.3700 ,  19.8900 ,  89.5200],
  [-24.8500 ,  18.0000 ,  89.0900],
  [-23.9600 ,  17.4800 ,  90.0800],
  [-24.6600 ,  17.2400 ,  87.7500],
  [-24.0800 ,  15.8500 ,  88.0100],
  [-23.9600 ,  15.1600 ,  86.7600],
  [-23.3400 ,  13.7100 ,  87.1000],
  [-21.9600 ,  13.8700 ,  87.6300],
  [-24.1800 ,  13.0300 ,  88.1100],
  [-23.2900 ,  12.8200 ,  85.7600],
  [-23.1900 ,  11.2800 ,  86.2200],
  [-21.8100 ,  11.0000 ,  86.7000],
  [-24.1500 ,  11.0300 ,  87.3200],
  [-23.5300 ,  10.3200 ,  84.9800],
  [-23.5400 ,   8.9800 ,  85.4800],
  [-23.8600 ,   8.0100 ,  84.3400],
  [-23.9800 ,   6.5760 ,  84.8900],
  [-23.2800 ,   6.4460 ,  86.1300],
  [-23.3000 ,   5.7330 ,  83.7800],
  [-22.7300 ,   4.5360 ,  84.3100],
  [-22.2000 ,   6.7130 ,  83.3000],
  [-22.7900 ,   8.0170 ,  83.3800],
  [-21.8100 ,   6.4120 ,  81.9200],
  [-20.8500 ,   5.5220 ,  81.5200],
  [-20.8300 ,   5.5670 ,  80.1200],
  [-21.7700 ,   6.4720 ,  79.7400],
  [-22.3400 ,   6.9680 ,  80.8000],
  [-20.0100 ,   4.6970 ,  82.1500],
  [-19.1800 ,   3.9390 ,  81.4700] ]);

if __name__=='__main__':

    import impetuous.clustering as impc

    distance_matrix = impc.absolute_coordinates_to_distance_matrix( coordinates )
    ordered_coordinates = impc.distance_matrix_to_absolute_coordinates( distance_matrix , n_dimensions=3 )

    print ( pd.DataFrame(ordered_coordinates).T )

```

You will notice that the largest variation is now aligned with the `X axis`, the second most variation aligned with the `Y axis` and the third most, aligned with the `Z axis` while the graph topology remained unchanged.

# Example 7: Retrieval and analysis of obesity data

In this example, we will show an analysis similar to the one conducted in Example 4. The only difference here is that we will model all of the data present in the journal. This includes the simultaneous analysis of categorical and number range descriptors present in the journal. We use an [impetuous shell](https://github.com/richardtjornhammar/rixcfgs/blob/master/code/environments/impetuous-shell.nix) and download the required [python file](https://gist.github.com/richardtjornhammar/1b9f5742391b1bcf30f4821a00f30b6a) and execute it in the shell. Now you are done! Was that too fast? ok, so what is this about?

You will see that the python code downloads a data directory (if you're using GNU/Linux), extracts it, curates it and performs the analysis. The directory contains sample data with information about both the platform and the sample properties. In our case a sample can come from any of `6` different platforms and belong to either `lean` or `obese` `females` or `males`. We collect the information and skip all but the `GPL8300` platform data. Now we have a journal that describes how well the sample was collected (with integer value ranges) and the sample categories as well as gene transcripts belonging to the samples. We can see that the common property for all samples are that they all are dealing with `obesity`, `adipocyte`, `inflammation` and `gene expression`. The journal now has the form


|      | GSM47229 | GSM47230 | GSM47231 | GSM47232 | ... | GSM47334 | GSM47335 | GSM47336 | GSM47337 |
|:---  |      ---:|      ---:|      ---:|      ---:|:---:|      ---:|      ---:|      ---:|      ---:|
|C(Array)|       HG_U95Av2 |   HG_U95Av2 |   HG_U95Av2 |   HG_U95Av2 | ... |  HG_U95Av2 |  HG_U95Av2 |  HG_U95Av2 |  HG_U95Av2|
|C(Types)|     lean-female | lean-female | lean-female | lean-female | ... | obese-male | obese-male | obese-male | obese-male|
|C(Type0)|            lean |        lean |        lean |        lean | ... |      obese |      obese |      obese |      obese|
|C(Type1)|          female |      female |      female |      female | ... |       male |       male |       male |       male|
|C(Platform)|      GPL8300 |     GPL8300 |     GPL8300 |     GPL8300 | ... |    GPL8300 |    GPL8300 |    GPL8300 |    GPL8300|
|Marginal   |          355 |         340 |         330 |         362 | ... |        357 |        345 |        377 |        343|
|Present    |         5045 |        5165 |        5581 |        4881 | ... |       4355 |       4911 |       5140 |       5672|
|Absent     |         7225 |        7120 |        6714 |        7382 | ... |       7913 |       7369 |       7108 |       6610|
|NoCall     |            0 |           0 |           0 |           0 | ... |          0 |          0 |          0 |          0|

Since we put extra effort into denoting all categoricals with `C( )` we can solve the problem for the entire journal in one go with

```
formula = 'f~'+'+'.join(journal_df.index.values)
```
which becomes
```
f~C(Array)+C(Types)+C(Type0)+C(Type1)+C(Platform)+Marginal+Present+Absent+NoCall
```
and the final analysis of the data becomes exceptionally simple, again by writing
```
    from impetuous.quantification import multifactor_evaluation
    multifactor_results = multifactor_evaluation ( analyte_df , journal_df , formula )
    multifactor_results.to_excel('obesity_transcripts.xlsx')
```
Now we can see which transcripts are sensitive to the numerical quality measures as well as the categorical instances that we might be interested in. Take for example the genes that seem to regulate obesity
```
np.array([['HSPA1A','HSPA1B', 'HSPA1L', 'IGFBP7', 'TMSB10', 'TMSB4X', 'RPLP2',
        'SNORA52', 'COL3A1', 'CXCL12', 'FLNA', 'AGPAT2', 'GPD1', 'ACTB',
        'ACTG1', 'RARRES2', 'COL6A2', 'HSPB6', 'CLU', 'TAGLN', 'HLA-DRA',
        'PFKFB3', 'MAOB', 'DPT', 'NQO1', 'S100A4', 'LIPE', 'CCND1',
        'FASN', 'COL6A1', 'NOTCH3', 'PFKFB3'],
       ['ECM2', 'C1S', 'GLUL', 'ENPP2', 'PALLD', 'MAOA', 'B2M', 'SPARC',
        'HTRA1', 'CCL2', 'ACTB', 'AKR1C1', 'AKR1C2', 'LOC101930400',
        'EIF4A2', 'MIR1248', 'SNORA4', 'SNORA63', 'SNORA81', 'SNORD2',
        'PTPLB', 'GAPDH', 'CCL2', 'SAT1', 'IGFBP5', 'AES', 'PEA15',
        'ADH1B', 'PRKAR2B', 'PGM1', 'GAPDH','S100A10']], dtype=object)
```
which account for the top `64` obesity transcripts. We note that some of these are shared with diabetics. If we study which ones describes the `Marginal` or `Absent` genes we can see that there are some that we might want to exclude for technical reasons. We will leave that excercise for the curious reader.

# Notes

These examples were meant as illustrations of some of the codes implemented in the impetuous-gfa package. The impetuous visualisation codes requires [Bokeh](https://docs.bokeh.org/en/latest/index.html) and are still being migrated to work with the latest Bokeh versions.

# Manually updated code backups for this library :

GitLab:	https://gitlab.com/richardtjornhammar/impetuous

CSDN:	https://codechina.csdn.net/m0_52121311/impetuous



