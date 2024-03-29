# A Statistical Learning library for Humans
This toolkit currently offers enrichment analysis, hierarchical enrichment analysis, novel PLS regression, shape alignment, connectivity clustering, clustering and hierarchical clustering as well as factor analysis methods. The fine grained data can be studied via a statistical tests that relates it to observables in a coarse grained journal file. The final p values can then be rank corrected. 

Several novel algorithms have been invented as of this repository by the [author](https://richardtjornhammar.github.io/). Some of the algorithms rely on old scientific litterature, but still consitutes new/novel code implementations. 

These novel algorithms include but are not limited to:
* A graph construction and graph searching class can be found in src/impetuous/convert.py (NodeGraph). It was developed and invented as a faster alternative for hierarchical DAG construction and searching.
* A DBSCAN method utilizing [my](https://richardtjornhammar.github.io/) connectivity code as invented during my PhD.
* Hierarchical enrichment routine with conservative or lax extinction of evidence already accounted for. Used for multiple hypothesis testing.
* A q-value method for rank correcting p-values. The computation differs from other methods.
* A NLP pattern matching algorithm useful for sequence alignment clustering
* An tensor field optimisation code.
* High dimensional alignment code for aligning models to data.
* An SVD based variant of the Distance Geometry algorithm. For going from relative to absolute coordinates.
* A numpy implementation of Householder decomposition.
* A matrix diagonalisation algorithm. (Native SVD algorithm that is slow)
* A MultiFactorAnalysis class for on-the-fly fast evaluation of matrix to matrix relationships
* Rank reduction for group expression methods.
* Visualisation/JS plots via bokeh.
* Fibonacci sequence relationship
* Prime number assessment

[![License](https://img.shields.io/github/license/Qiskit/qiskit.svg?)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5109938.svg)](https://doi.org/10.5281/zenodo.5109938)
[![Downloads](https://pepy.tech/badge/impetuous-gfa)](https://pepy.tech/project/impetuous-gfa)

Visit the active code via :
https://github.com/richardtjornhammar/impetuous

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

In order to run the code in this notebook you must enter a sensible working environment. Don't worry! We have created one for you. It's version controlled against python3.7 (and python3.8) and you can get the file here:

https://github.com/richardtjornhammar/rixcfgs/blob/master/code/environments/impetuous-shell.nix

Since you have installed Nix as well as WSL, or use a Linux (NixOS) or bsd like system, you should be able to execute the following command in a termnial:

```
$ nix-shell impetuous-shell.nix
```

Now you should be able to start your jupyter notebook locally:

```
$ jupyter-notebook impetuous.ipynb
```

and that's it.

# Test installation
You can [download](https://gist.githubusercontent.com/richardtjornhammar/e2f95f70c3ba56e764117aa0f7398dfb/raw/e9d85c22b9ba6264d1a19cfab1f00d7f7b3f62ef/test_impetuous.py)  and run that python file to verify the installation. If it isn't working then there is an error with the package 

# [Example 0](https://gist.githubusercontent.com/richardtjornhammar/34e163cba547d6c856d902244edc2039/raw/2a069b062df486b8d081c8cfedbbb30321e44f36/example0.py):
After installing `impetuous-gfa version >=0.66.5` you should be able to execute the code
```
if __name__=='__main__':
    import impetuous as imp
    import impetuous.hierarchical as imphi
    import impetuous.clustering as impcl
    import impetuous.fit as impfi
    import impetuous.pathways as imppa
    import impetuous.visualisation as impvi
    import impetuous.optimisation as impop
    import impetuous.convert as impco
    import impetuous.probabilistic as imppr
    import impetuous.quantification as impqu
    import impetuous.spectral as impsp
    import impetuous.reducer as impre
    import impetuous.special as impspec
```
You can execute it easily when you are in the [impetuous environment](https://github.com/richardtjornhammar/rixcfgs/blob/master/code/environments/impetuous-shell.nix). Just write
```
$ wget https://gist.githubusercontent.com/richardtjornhammar/34e163cba547d6c856d902244edc2039/raw/2a069b062df486b8d081c8cfedbbb30321e44f36/example0.py
$ python3 example0.py
```
And if it doesn't work then contact [me](https://richardtjornhammar.github.io/) and I'll try and get back within 24h

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

# [Example 3](https://gist.githubusercontent.com/richardtjornhammar/78f3670ea406e1e2e8e244b6fbc31f2c/raw/a34577fa87234867cda385cb26dbf72aa266bac6/example3.py): Novel NLP sequence alignment

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

# [Example 4](https://gist.githubusercontent.com/richardtjornhammar/a9704b238c74080fdea0827608a10a9a/raw/277ca835b8c56c3bb25d21e28e0d0eaa1661201f/example4.py): Diabetes analysis

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

# [Example 5](https://gist.githubusercontent.com/richardtjornhammar/ad932891349ee1534050fedb766ac5e3/raw/0cf379b6b94f92ea12acab72f84ba30f7b8860ad/example5.py): Understanding what it means

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

# [Example 6](https://gist.githubusercontent.com/richardtjornhammar/b1b71fb5669425a8b52c9bc6b530c418/raw/4f21b22b9b85bed2a387101a7b234320024abee2/example6.py): Absolute and relative coordinates

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

# [Example 7](https://gist.github.com/richardtjornhammar/1b9f5742391b1bcf30f4821a00f30b6a): Retrieval and analysis of obesity data

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

# [Example 8](https://gist.githubusercontent.com/richardtjornhammar/5bac33de1497bd3a1117d709b672d918/raw/96dbb65876c2f742b1c6a27e502be006416fd99e/example8.py): Latent grouping assumptions. Building a Parent-Child list

So you are sitting on a large amount of groupings that you have a significance test for. Testing what you are interested in per analyte symbol/id. Since you will conduct a large amount of tests there is also a large risk that you will technically test the same thing over and over again. In order to remove this effect from your group testing you could employ my `HierarchicalEnrichment` routine, but then you would also need a relationship file describing how to build a group DAG Hierarchy. This can be done with a relationship file that contains a `parent id`, a `tab delimiter` and a `child id` on each line. The routine that I demonstrate here uses a divide-and-conquer type approach to construct that information, which means that a subgroup, or child, is only assigned if it is fully contained within the parents definition. You can create redundant assignments by setting `bSingleDescent=False`, but it is not the recommended default setting.

Construction of the downward node relationships can be done with my `build_pclist_word_hierarchy` routine. Let us assume that you are sitting on the following data:
```
    portfolios = { 'PORT001' : ['Anders EQT' ,['AAPL','GOOG','IBM','HOUSE001','OTLY','GOLD','BANANAS'] ],
                   'PORT002' : ['Anna EQT'   ,['AAPL','AMZN','HOUSE001','CAR','BOAT','URANIUM','PLUTONIUM','BOOKS'] ],
                   'PORT003' : ['Donald EQT' ,['EGO','GOLF','PIES','HOUSE100','HOUSE101','HOUSE202'] ] ,
                   'PORT004' : ['BOB EQT'    ,['AAPL','GOOG'] ],
                   'PORT005' : ['ROB EQT'    ,['AMZN','BOOKS'] ],
                   'PORT006' : ['KIM EQT'    ,['URANIUM','PLUTONIUM'] ],
                   'PORT007' : ['LIN EQT'    ,['GOOG'] ] }
```
Then you might have noticed that some of the portfolios seem to contain the others completely. In order to derive the direct downward relationship you can issue the following commands (after installing `impetuous version>=0.64.1`
```
    import impetuous.hierarchical as imph
    pclist = imph.build_pclist_word_hierarchy ( ledger = portfolios , group_id_prefix='PORT' , root_name='PORT000')
```
which will return the list you need. You can now save it as a node relationship file and use that in my DAG construction routine.

Lets instead assume that you want to read the analyte groupings from a [file](https://gist.githubusercontent.com/richardtjornhammar/6780e6d99e701fcc83994cc7a5f77759/raw/c37eaeeebc4cecff200bebf3b10dfa57984dbb84/new_compartment_genes.gmt), then you could issue :
```
    import os
    os.system('wget https://gist.githubusercontent.com/richardtjornhammar/6780e6d99e701fcc83994cc7a5f77759/raw/c37eaeeebc4cecff200bebf3b10dfa57984dbb84/new_compartment_genes.gmt')
    filename = 'new_compartment_genes.gmt'
    pcl , pcd = imph.build_pclist_word_hierarchy ( filename = filename , bReturnList=True )
```
If there are latent assumptions for some groupings then you can read them out by checking what the definitions refers to (here we already know that there is one for the mitochondrion definition):
```
    for item in pcl :
        if  'mito' in pcd[item[1]][0] or 'mela' in pcd[item[1]][0] :
            print ( pcd[item[0]][0] , ' -> ' , pcd[item[1]][0] )
```
which will tell you that
```
full cell  ->  melanosome membrane
full cell  ->  mitochondrial inner membrane
full cell  ->  mitochondrial matrix
melanosome membrane   ->  mitochondrion
full cell  ->  mitochondrial outer membrane
full cell  ->  mitochondrial intermembrane space
```
the definition for the mitochondrion is fully contained within the melanosome membrane definition and so testing that group should try and account for the mitochondrion. This can be done with the `HierarchicalEnrichment` routine exemplified above. We know that the melanosome membrane is associated with sight and that being diabetic is associated with mitochondrial dysfunction, but also that diabetic retinopathy affects diabetics. We see here that there is a knowledge based genetic connection relating these two spatially distinct regions of the cell.

# [Example 9](https://gist.githubusercontent.com/richardtjornhammar/e84056e0b10f8d550258a1e8944ee375/raw/e44e7226b6cb8ca486ff539ccfa775be981a549c/example9.py): Impetuous [deterministic DBSCAN](https://github.com/richardtjornhammar/impetuous/blob/master/src/impetuous/clustering.py) (search for dbscan)

[DBSCAN](https://en.wikipedia.org/wiki/DBSCAN) is a clustering algorithm that can be seen as a way of rejecting points, from any cluster, that are positioned in low dense regions of a point cloud. This introduces holes and may result in a larger segment, that would otherwise be connected via a non dense link to become disconnected and form two segments, or clusters. The rejection criterion is simple. The central concern is to evaluate a distance matrix <img src="https://render.githubusercontent.com/render/math?math=D_{ij}">  with an applied cutoff <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> this turns the distances into true or false values depending on if a pair distance between point i and j is within the distance cutoff. This new binary Neighbour matrix <img src="https://render.githubusercontent.com/render/math?math=N_{ij}=D_{ij}\le\epsilon"> tells you wether or not two points are neighbours (including itself). The DBSCAN criterion states that a point is not part of any cluster if it has fewer than `minPts` neighbors. Once you've calculated the distance matrix you can immediately evaluate the number of neighbors each point has and the rejection criterion, via <img src="https://render.githubusercontent.com/render/math?math=R_i=(\sum_{j} D_{ij}\le\epsilon)-1 < minPts">. If the rejection vector R value of a point is True then all the pairwise distances in the distance matrix of that point is set to a value larger than epsilon. This ensures that a distance matrix search will reject those points as neighbours of any other for the choosen epsilon. By tracing out all points that are neighbors and assessing the [connectivity](https://github.com/richardtjornhammar/impetuous/blob/master/src/impetuous/clustering.py) (search for connectivity) you can find all the clusters.

In this [example](https://gist.githubusercontent.com/richardtjornhammar/e84056e0b10f8d550258a1e8944ee375/raw/e44e7226b6cb8ca486ff539ccfa775be981a549c/example9.py) we do exactly this for two gaussian point clouds. The dbscan search is just a single line `dbscan ( data_frame = point_cloud_df , eps=0.45 , minPts=4 )`, while the last lines are there to plot the [results](https://richardtjornhammar.github.io/?https://gist.githubusercontent.com/richardtjornhammar/0cc0ff037e88c76a9d65387155674fd1/raw/f8c740cd648247889f21eeaadb7b7c74577540be/index.html) ( has [graph revision dates](https://gist.github.com/richardtjornhammar/0cc0ff037e88c76a9d65387155674fd1/revisions) )

The [radial distribution function](https://en.wikipedia.org/wiki/Radial_distribution_function) is a useful tool for visualizing whether or not there are radial clustering tendencies at any average distance between the group of interest and any other constituents of the system. This structure assessment method is usually used for [analysis](https://gist.githubusercontent.com/richardtjornhammar/33162d3be1e92f1b1fafbd9e46954e91/raw/c0685bb79527c947213ffe08973d3ea4e072257e/argon.py) of particle systems, i.e. see [liquid structure](
https://richardtjornhammar.github.io/?https://gist.githubusercontent.com/richardtjornhammar/bc1e9a8b4c693a338ef812a74ab685e9/raw/5276ce75960fa99b5a80972b9e187dc2df29831b/index.html). It is implemented in the `clustering` module and is demonstrated [here](https://gist.githubusercontent.com/richardtjornhammar/f25ec2eef0703f07ebc0d678123f450e/raw/b9ac597a9d2587727af3cb06a8090ad0eaf0ba49/example10.py). If there is a significant density close to `r=0` then you cannot separate the group from the mean background. This also means that any significance test between those groups will tell you that the grouping is insignificant. The [resulting plot](https://richardtjornhammar.github.io/?https://gist.githubusercontent.com/richardtjornhammar/ff417450790c8c885b077fc7ee20409d/raw/65554165ffddf309272a14d6aba1e3fac9fa1a13/index.html) has [revision dates](https://gist.github.com/richardtjornhammar/ff417450790c8c885b077fc7ee20409d/revisions). Since the radial distribution function calculates the spherically symmetric distribution of points surrounding an analyte, or analyte group, of interest it is effectively analogous to segmenting the distance matrix and leaving out any self interaction distances that may or may not be present. 

The functions `select_from_distance_matrix` uses boolean indexing to select rows and columns (it is symmetric) in the distance matrix and the `exclusive_pdist` function calculates all pairs between the points in the two separate groups.


# Example 10: Householder decomposition

In this example we will compare the decompostion of square and rectangular matrices before and after Householder decomposition. We recall that the Householder decomposition is a way of factorising matrices into orthogonal components and a tridiagonal matrix. The routine is implemented in the `impetuous.reducer` module under the name `Householder_reduction`. Now, why is any of that important? The Householder matrices are deterministically determinable and consitutes an unambigous decomposition of your data. The factors are easy to use to further solve what different types of operations will do to your original matrix. One can, for instance, use it to calculate the ambigous SVD decomposition or calculate eigenvalues for rectangular matrices.

Let us assume that you have a running environment and a set of matrices that you like
```
import numpy as np
import pandas as pd

if __name__=='__main__' :
    from impetuous.reducer import ASVD, Householder_reduction

    df = lambda x:pd.DataFrame(x)
    if True :
        B = np.array( [ [  4 ,  1 , -2 ,  2 ] ,
                   [  1 ,  2 ,  0 ,  1 ] ,
                   [ -2 ,  0 ,  3 , -2 ] ,
                   [  2 ,  1 , -2 , -1 ] ] )

    if True :
        A = np.array([ [ 22 , 10 ,  2 ,   3 ,  7 ] ,
                    [ 14 ,  7 , 10 ,   0 ,  8 ] ,
                    [ -1 , 13 , -1 , -11 ,  3 ] ,
                    [ -3 , -2 , 13 ,  -2 ,  4 ] ,
                    [  9 ,  8 ,  1 ,  -2 ,  4 ] ,
                    [  9 ,  1 , -7 ,   5 , -1 ] ,
                    [  2 , -6 ,  6 ,   5 ,  1 ] ,
                    [  4 ,  5 ,  0 ,  -2 ,  2 ] ] )
```
you might notice that the eigenvalues and the singular values of the square matrix `B` look similar
```
    print ( "FOR A SQUARE MATRIX:" )
    print ( "SVD DIAGONAL MATRIX ",df(np.linalg.svd(B)[1]) )
    print ( "SORTED ABSOLUTE EIGENVALUES ", df(sorted(np.abs(np.linalg.eig(B)[0]))[::-1]) )
    print ( "BOTH RESULTS LOOK SIMILAR" )
```
but that the eigenvalues for the Householder reduction of the matrix B and the matrix B are the same
```
    HB = Householder_reduction ( B )[1]
    print ( np.linalg.eig( B)[0]  )
    print ( np.linalg.eig(HB)[0]  )
```
We readily note that this is also true for the singular values of the matrix `B` and the matrix `HB`. For the rectangular matrix `A` the eigenvalues are not defined when using `numpy`. The `SVD` decomposition is defined and we use it to check if the singular values are the same for the Householder reduction of the matrix A and the matrix A.
```
    print ( "BUT THE HOUSEHOLDER REDUCTION IS")
    HOUSEH = Householder_reduction ( A )[1]
    print ( "SVD ORIGINAL  : " , df(np.linalg.svd(A)[1]) )
    print ( "SVD HOUSEHOLD : " , df(np.linalg.svd(HOUSEH)[1]) )
```
and lo and behold.
```
    n = np.min(np.shape(HOUSEH))
    print ( "SVD SKEW   H : " , df(np.linalg.svd(HOUSEH)[1]) )
    print ( "SVD SQUARE H : " , df(np.linalg.svd(HOUSEH[:n,:n])[1]) )
    print ( "SVD ORIGINAL : " , df(np.linalg.svd(A)[1]) )
    print ( "EIGENVALUES  : " , np.linalg.eig(HOUSEH[:n,:n])[0] )
```
They are. So we feel confident that using the eigenvalues from the square part of the Householder matrix (the rest is zero anyway) to calculate the eigenvalues of the rectangular matrix is ok. But wait, why are they complex valued now? :^D

We can also reconstruct the original data by multiplying together the factors of either decomposition
```
    F,Z,GT = Householder_reduction ( A )
    U,S,VT = ASVD(A)

    print ( np.dot( np.dot(F,Z),GT ) )
    print ( np.dot( np.dot(U,S),VT ) )
    print ( A )
```
Thats all for now folks!

# Example 11: The NodeGraph class for agglomerative hierarchical clustering

An alternative way of constructing a DAG hierarchy is by using distance matrix linkages.

```
import numpy as np
import typing

if __name__=='__main__' :

    import time
    from impetuous.clustering import linkage

    D = [[0,9,3,6,11],[9,0,7,5,10],[3,7,0,9,2],[6,5,9,0,8],[11,10,2,8,0] ]
    print ( np.array(D) )
    t0 = time.time()
    links = linkage( D, command='min')
    dt = time.time()-t0
    print ('min>', linkages( D, command='min') , dt) # SINGLE LINKAGE (MORE ACCURATE)
    print ('max>', linkages( D, command='max') )     # COMPLETE LINKAGE

    import impetuous.convert as gg

    GN = gg.NodeGraph()
    GN .linkages_to_graph_dag( links )
    GN .write_json( jsonfile='./graph_hierarchy.json' )

```

# Example 12: Use the NodeGraph class to create a gmt file

When your data is high dimensional one alternative to analysing it is via statistical methods based on groupings. One way of obtaining the groupings is by creating a DAG hierarchy. Here we do that and write the resulting information to `gmt` and `json` files. You can calculate pairwise correlation distances or any other distance matrix type that describes your data and pass it either to the linkage methods or the slower distance matrix conversion methods. In this case the two are equivalent and produces the same results. If you happen to have a list of names corresponding to the name of a analyte in the distance matrix then you can supply a dictionary to the `NodeGraph` class in order to translate the distance indices to their proper names.

```
import numpy   as np
import typing

if __name__=='__main__' :

    import time
    from impetuous.clustering import linkage

    D = np.array([[0,9,3,6,11],[9,0,7,5,10],[3,7,0,9,2],[6,5,9,0,8],[11,10,2,8,0] ])
    print ( np.array(D) )
    t0 = time.time()
    links = linkages( D, command='min')
    dt = time.time()-t0
    print ('min>', linkages( D, command='min') , dt) # SINGLE LINKAGE (MORE ACCURATE)

    import impetuous.convert as gg

    GN = gg.NodeGraph()
    GN .linkages_to_graph_dag( links )
    GN .write_json( jsonfile='./lgraph_hierarchy.json' )
    GN .rename_data_field_values( {0:'UNC13C',1:'PCYT2',2:'BDH1',3:'OMA1',4:'VEGFA'} , 'analyte ids' )
    GN .write_gmt( "./lgroups.gmt" )

    GD = gg.NodeGraph()
    GD .distance_matrix_to_graph_dag( D )
    GD .write_json( jsonfile='./draph_hierarchy.json' )
    GD .write_gmt( "./dgroups.gmt" )

```
Note that the rename method was called after we wrote the `json` hierarchy and thus only the `lgroups.gmt` contain the proper names while the other are annotated with the internal index values. Cluster names are deduced by the index values joined by a `.`. If you look in the `gmt` file with a text editor you will see that the first column contains the `child` cluster and the second columns first entry contains the `parent` cluster name (it is also followed by more information joined in with a `:`). The field delimiter for `gmt` file fields is a tab delimiter.

See also solution with less dependencies in the [graphtastic](https://github.com/richardtjornhammar/graphtastic) library

# [Example 13](https://gist.githubusercontent.com/richardtjornhammar/74175e415c4cf35af8424696589a57a7/raw/28b902ec282b43cbd0f5e34f6ceffea257d7e9a1/cexp.py): Compare distance geometry clustering with UMAP

This library contains several clustering algorithms and fitting procedures. In this example we will use the SVD based distance geometry algorithm to project the distance matrix of mnist digits onto a 2D surface and compare the result with what can be obtained using the [UMAP](https://umap-learn.readthedocs.io/en/latest/plotting.html) methods. UMAP works in a nonlinear fashion in order to project your data onto a surface that also maximizes mutual distances. Distance geometry works on nonlinear data described by a distance matrix, but creates a linear projection onto the highest variance dimensions in falling order. Note that distance geometry is not a PCA method but a transformation between relative distances and their absolute coordinates. UMAP can distort the topology of absolute coordinates while distance geometry does not. UMAP is however better at discriminating distinct points.

Lets have a look at the setup
```
import numpy as np

# https://umap-learn.readthedocs.io/en/latest/plotting.html
import sklearn.datasets
import umap

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors

matplotlib.use('TkAgg',force=True)

from impetuous.clustering import distance_matrix_to_absolute_coordinates , absolute_coordinates_to_distance_matrix
#
# COMPARING WITH MY DISTANCE GEOMETRY SOLUTION
#
```

We have now installed both the `impetuous-gfa` as well as the `umap-learn` libraries. So we load the data and prepare the colors we want to use

```
if __name__ == '__main__' :

    pendigits = sklearn.datasets.load_digits()
    targets   = pendigits.target

    all_colors = list ( mcolors.CSS4_COLORS.keys() )
    NC = len ( all_colors )
    NU = len ( set( targets ) )
    plot_colors = [ all_colors[ic] for ic in  [ int(np.floor(NC*(t+0.5)/NU)) for t in targets ] ]
```

Now we project our data using both methods

```
    #
    # DISTANCE GEOMETRY CLUSTERING FOR DIMENSIONALITY REDUCTION
    distm      = absolute_coordinates_to_distance_matrix ( pendigits['data'] )
    projection = distance_matrix_to_absolute_coordinates ( distm , n_dimensions = 3 )
    #
    # UMAP CLUSTERING FOR DIMENSIONALITY REDUCTION
    umap_crds  = umap.UMAP().fit_transform( pendigits.data )
```
Now we want to plot the results with `matplotlib`
```
    fig, axs = plt.subplots( 1, 2, figsize=(20, 20) )
    
    axs[0].scatter( umap_crds[:, 0] , umap_crds[:, 1] ,
                    c=plot_colors , marker='.', alpha=1. , s=1. )

    for x,y,c in zip ( projection[0], projection[1], plot_colors ) :
        axs[1].plot ( x, y , c , marker='.' )

    plt.show()
```
and finally save the image as an `svg`
```
    image_format = 'svg'
    image_name = 'myscatter_comparison.svg'
    fig.savefig(image_name, format=image_format, dpi=300)

```
It is readily viewable below and we can see that the UMAP and Distance Geometry algorithms both cluster the data. But that the UMAP was able to discriminate better, forcing the solution into tighter clusters. Some of the clusters in the right hand side figure however separate in the third dimension (not shown).

![teaser](https://gist.githubusercontent.com/richardtjornhammar/997e179a5e773fe054d2f8edcdb5cd72/raw/78ec4e67ffa4585918ca7e117c5a080b6188447b/myscatter_comparison.svg)

# Example 14: Connectivity, hierarchies and linkages

In the `impetuous.clustering` module you will find several codes for assessing if distance matrices are connected at some distance or not. `connectivity` and `connectedness` are two methods for establishing the number of clusters in the binary Neighbour matrix. The Neighbour matrix is just the pairwise distance between the parts `i` and `j` of your system (<img src="https://render.githubusercontent.com/render/math?math=D_{ij}">) with an applied cutoff (<img src="https://render.githubusercontent.com/render/math?math=N_{ij}=D_{ij}\le\epsilon">) and is related to the adjacency matrix from graph theory by adding an identity matrix to the adjacency matrix (<img src="https://render.githubusercontent.com/render/math?math=A_{ij}=N_{ij} - I_{ij}">). The three boolean matrices that describe a system at some distance cutoff (<img src="https://render.githubusercontent.com/render/math?math=\epsilon">) are: the Identity matrix (<img src="https://render.githubusercontent.com/render/math?math=I_{ij} = D_{ij}\equiv0 ">), the Adjacency matrix (<img src="https://render.githubusercontent.com/render/math?math=A_{ij}= D_{ij}\le\epsilon - I_{ij}">) and the Community matrix (<img src="https://render.githubusercontent.com/render/math?math=C_{ij}=D_{ij}>\epsilon">). We note that summing the three matrices will return `1` for any `i,j` pair. 

"Connection" algorithms, such as the two mentioned, evaluate every distance and add them to the same cluster if there is any true overlap for a specific distance cutoff. ["Link" algorithms](https://online.stat.psu.edu/stat555/node/85/) try to determine the number of clusters for all unique distances by reducing and ignoring some connections to already linked constituents of the system in accord with a chosen heuristic. 

The "Link" codes are more efficient at creating a link hierarchy of the data but can be thought of as throwing away information at every linking step. The lost information is deemed unuseful by the heuristic. The full link algorithm determines the new cluster distance to the rest of the points in a self consistent fashion by employing the same heuristic. Using simple linkage, or `min` value distance assignment, will produce an equivalent [hierarchy](https://online.stat.psu.edu/stat555/node/86/) as compared to the one deduced by a connection algorithm. Except for some of the cases when there are distance ties in the link evaluation. This is a computational quirk that does not affect "connection" based hierarchy construction.

The "Link" method is thereby not useful for the deterministic treatment of a particle system where all the true connections in it are important, such as in a water bulk system when you want all your quantum-mechanical waters to be treated at the same level of theory based on their connectivity at a specific level or distance. This is indeed why my connectivity algorithm was invented by me in 2009. If you are only doing black box statistics on a complete hierarchy then this distinction is not important and computational efficiency is probably what you care about. You can construct hierarchies from both algorithm types but the connection algorithm will always produce a unique and well-determined structure while the link algorithms will be unique but structurally dependent on how ties are resolved and which heuristic is employed for construction. The connection hierarchy is exact and deterministic, but slow to construct, while the link hierarchies are heuristic dependent, but fast to construct. We will study this more in the following code example as well as the case when they are equivalent.

##  14.1 Link hierarchy construction
The following code produces two distance matrices. One has distance ties and the other one does not. The second matrix is well known and the correct minimal linkage hierarchy is well known. Lets see compare the results between scipy and our method.
```
import numpy as np
from impetuous.clustering import absolute_coordinates_to_distance_matrix
from impetuous.clustering import linkages, scipylinkages
from impetuous.special import lint2lstr

if __name__ == '__main__' :
    
    xds = np.array([ [5,2],
                   [8,4],
                   [4,6],
                   [3,7],
                   [8,7],
                   [5,10]
                  ])

    tied_D = np.array([ np.sum((p-q)**2) for p in xds for q in xds ]).reshape(len(xds),len(xds))

    print ( tied_D )
    lnx1 = linkages ( tied_D.copy() , command='min' )
    lnx2 = scipylinkages(tied_D,'min')

    print ( '\n',lnx1 ,'\n', lnx2 )
    
    D = np.array([[0,9,3,6,11],[9,0,7,5,10],[3,7,0,9,2],[6,5,9,0,8],[11,10,2,8,0] ])

    print ('\n', np.array(D) )

    lnx1 = linkages ( D , command='min' )
    lnx2 = scipylinkages( D,'min')

    print ( '\n',lnx1 ,'\n', lnx2 )
```

We study the results below
```
[[ 0 13 17 29 34 64]
 [13  0 20 34  9 45]
 [17 20  0  2 17 17]
 [29 34  2  0 25 13]
 [34  9 17 25  0 18]
 [64 45 17 13 18  0]]

 {'2.3': 2, '1.4': 9.0, '1.4.0': 13.0, '2.3.5': 13.0, '2.3.5.1.4.0': 17.0, '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0} 
 {'1': 2.0, '4': 2.0, '0': 2.0, '2.3': 2.0, '5': 2.0, '1.4': 9.0, '0.1.4': 13.0, '2.3.5': 13.0, '0.1.2.3.4.5': 17.0}

 [[ 0  9  3  6 11]
 [ 9  0  7  5 10]
 [ 3  7  0  9  2]
 [ 6  5  9  0  8]
 [11 10  2  8  0]]

 {'2.4': 2, '2.4.0': 3.0, '1.3': 5.0, '1.3.2.4.0': 6.0, '0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
 {'2.4': 2.0, '0': 2.0, '1': 2.0, '3': 2.0, '0.2.4': 3.0, '1.3': 5.0, '0.1.2.3.4': 6.0}
```
We see that the only difference for these two examples are how the unclustered indices are treated. In our method they are set to the identity distance value of zero while scipy attributes them the lowest non diagonal value in the distance matrix.

## 14.2 Connectivity hierarchy construction
Now we employ the `connectivity` algorithm for construction of the hierarchy. In the below code segment the first loop calls the function directly and the second calls the `impetuous.hierarchy_matrix` function
```
    import impetuous.hierarchical as imph
    from impetuous.clustering import connectivity

    unique_distances = sorted(list(set(D.reshape(-1))))
    for u in unique_distances :
        results = connectivity(D,u)
        print ( u , results )
        if len(results[0]) == 1 :
            break

    res = imph.hierarchy_matrix ( D )
    print ( res )
```
with the results
```
0 ([1, 1, 1, 1, 1], array([[0, 0],
       [1, 1],
       [2, 2],
       [3, 3],
       [4, 4]]))
2 ([1, 1, 1, 2], array([[0, 0],
       [1, 1],
       [3, 2],
       [2, 3],
       [3, 4]]))
3 ([1, 1, 3], array([[2, 0],
       [0, 1],
       [2, 2],
       [1, 3],
       [2, 4]]))
5 ([2, 3], array([[1, 0],
       [0, 1],
       [1, 2],
       [0, 3],
       [1, 4]]))
6 ([5], array([[0, 0],
       [0, 1],
       [0, 2],
       [0, 3],
       [0, 4]]))
{'hierarchy matrix':(array([[0, 1, 2, 3, 4],
       [0, 1, 3, 2, 3],
       [2, 0, 2, 1, 2],
       [1, 0, 1, 0, 1],
       [0, 0, 0, 0, 0]]),'lookup':{0: [0, 0, 1.0], 1: [1, 2, 1.25], 2: [2, 3, 1.6666666666666667], 3: [3, 5, 2.5], 4: [4, 6, 5.0]}}
```
and we see that the system has 5 unique levels. The hierarchy matrix increase in distance as you traverse down. The first row is level `0` with distance `0` and all items are assigned to each own cluster. The third row, level `2`, contains three clusters at distance `3` and the three clusters are `0.2.4` as well as `1` and `3`. We see that they become joined at level `3` corresponding to distance `5`.

The final complete clustering results can be obtained in this alternative way for the `connectivity` hierarchy
```
    print ( imph.reformat_hierarchy_matrix_results ( res['hierarchy matrix'],res['lookup'] ) )
```
with the result
```
{(0,): 0, (1,): 0, (2,): 0, (3,): 0, (4,): 0, (2, 4): 2, (0, 2, 4): 3, (1, 3): 5, (0, 1, 2, 3, 4): 6}
```
which is well aligned with the previous results, but the `connectivity` approach is slower to employ for constructing a hierarchy.

## Comparing hierarchies of an equidistant plaque

We know by heart that a triagonal mesh with a link length of one is fully connected at only that distance. So lets study what the hierarchical clustering results will yield.
```
    def generate_plaque(N) :
        L,l = 1,1
        a  = np.array( [l*0.5, np.sqrt(3)*l*0.5] )
        b  = np.array( [l*0.5,-np.sqrt(3)*l*0.5] )
        x_ = np.linspace( 1,N,N )
        y_ = np.linspace( 1,N,N )
        Nx , My = np.meshgrid ( x_,y_ )
        Rs = np.array( [ a*n+b*m for n,m in zip(Nx.reshape(-1),My.reshape(-1)) ] )
        return ( Rs )
        
    from clustering import absolute_coordinates_to_distance_matrix as c2D
    D = c2D( generate_plaque(N=3))
    #
    # CONNECTIVITY CONSTRUCTION
    print ( imph.reformat_hierarchy_matrix_results ( *imph.hierarchy_matrix( D ).values() ) )
    #
    # SCIPY LINKAGE CONSTRUCTION
    print ( scipylinkages(D,'min',bStrKeys=False) )
```
which readily tells us that
```
{(0,): 0.0, (1,): 0.0, (2,): 0.0, (3,): 0.0, (4,): 0.0, (5,): 0.0, (6,): 0.0, (7,): 0.0, (8,): 0.0, (0, 1, 3, 4): 0.9999999999999999, (2, 5): 0.9999999999999999, (6, 7): 0.9999999999999999, (0, 1, 2, 3, 4, 5, 6, 7, 8): 1.0}

{(6, 7): 0.9999999999999999, (0, 1, 3, 4): 0.9999999999999999, (2, 5): 0.9999999999999999, (8,): 0.9999999999999999, (0, 1, 2, 3, 4, 5, 6, 7, 8): 1.0}
```
and we see that everything is connected at the distance `1` and that the numerical treatment seems to have confused both algorithms in a similar fashion, but that `scipy` is assigning single index clusters the distance `1`

we measure the time it takes for both to complete ever large meshes
```
    from clustering import absolute_coordinates_to_distance_matrix as c2D
    T = []
    for N in range(3,40,2):
        D = c2D( generate_plaque(N=N))
        t0=time.time()
        r1= imph.reformat_hierarchy_matrix_results ( *imph.hierarchy_matrix( D ).values() )
        t1=time.time()
        r2= scipylinkages(D,'min',bStrKeys=False)
        t2=time.time()
        if N>2:
            T.append([N,t1-t0,t2-t1])

    for t in T:
        print(t)
```
and find the timing to be:
```
[4, 0.00019979476928710938, 0.0009992122650146484]
[6, 0.00045108795166015625, 0.003519296646118164]
[8, 0.0009257793426513672, 0.00949406623840332]
[10, 0.001996755599975586, 0.021444082260131836]
[12, 0.003604412078857422, 0.04308891296386719]
[14, 0.006237030029296875, 0.0793461799621582]
[16, 0.010350704193115234, 0.13524317741394043]
[18, 0.015902042388916016, 0.2159280776977539]
[20, 0.030185699462890625, 0.3255939483642578]
[22, 0.03534746170043945, 0.47675514221191406]
[24, 0.07047271728515625, 0.67844557762146]
[26, 0.06810998916625977, 0.929694652557373]
[28, 0.13647937774658203, 1.2459971904754639]
[30, 0.12457752227783203, 1.705310583114624]
[32, 0.1785578727722168, 2.111368417739868]
[34, 0.3048675060272217, 2.662834644317627]
[36, 0.27133679389953613, 3.3377525806427]
[38, 0.34802937507629395, 4.12202787399292]
```
So it is clear that a linkage method is more efficient for constructing complete hierarchies while a single `connectivity` calculation might be faster if you only want the clusters at a predetermined distance. Because in that case you don't need to calculate the entire hierarchy.

# Notes

These examples were meant as illustrations of some of the codes implemented in the impetuous-gfa package. 

The impetuous visualisation codes requires [Bokeh](https://docs.bokeh.org/en/latest/index.html) and are still being migrated to work with the latest Bokeh versions. For an example of the dynamic `triplot` routine (you can click on the lefthand and bottom scatter points) you can view it [here](https://rictjo.github.io/?https://gist.githubusercontent.com/richardtjornhammar/e6a7570140bd1216a5681c9d96ff157a/raw/6c9230f06b98c1226c6de455816a381c140236c8/index.html) ( with [revision dates](https://gist.github.com/richardtjornhammar/e6a7570140bd1216a5681c9d96ff157a/revisions) or download it [here](https://gist.github.com/richardtjornhammar/e6a7570140bd1216a5681c9d96ff157a/) ).

Some of the algorithms rely on the SVD implementation in Numpy. A switch is planned for the future.

# Manually updated code backups for this library :

GitLab:	https://gitlab.com/richardtjornhammar/impetuous

CSDN:	https://codechina.csdn.net/m0_52121311/impetuous

Bitbucket:	https://bitbucket.org/richardtjornhammar/impetuous
