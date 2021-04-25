# A Statistical Learning library for Humans
Decomposes a set of expressions into a group expression. The toolkit currently offers enrichment analysis, hierarchical enrichment analysis, PLS regression, Shape alignment or clustering as well as  rudimentary factor analysis.

The expression regulation can be studied via a statistical test that relates it to the observables in the journal file. The final p values are then FDR corrected and the resulting adjusted p values are produced.

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

# Usage example 1 : elaborate informatics

code: https://gitlab.com/stochasticdynamics/eplsmta-experiments
docs: https://arxiv.org/pdf/2001.06544.pdf

# Usage example 2 : simple regression code

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

# Usage example 3 : Novel NLP sequence alignment

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

# Manually updated code backups for this library :

GitLab:	https://gitlab.com/richardtjornhammar/impetuous

CSDN:	https://codechina.csdn.net/m0_52121311/impetuous

