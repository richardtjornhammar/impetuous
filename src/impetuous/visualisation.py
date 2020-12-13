import numpy as np
import pandas as pd
import bokeh

description="""Rudimentary plotting wrappers for impetuous using bokeh"""

run_in_notebook="""
    from bokeh.plotting import output_notebook, show
    output_notebook()"""

typecheck = lambda x,typ: typ in str(type(x))
list_typecheck = lambda xl,typ, func: func( [ typecheck(x,typ) for x in xl ] )

nice_colors = list( set( [ '#c5c8c6' , '#1d1f21' , '#282a2e' , '#373b41' , '#a54242' , '#cc6666' ,
                '#8c9440' , '#b5bd68' , '#de935f' , '#f0c674' , '#5f819d' , '#81a2be' ,
                '#85678f' , '#b294bb' , '#5e8d87' , '#8abeb7' , '#707880' ] ) )

def bscatter( X , Y , additional_dictionary=None , title='' , color='#ff0000' , p=None, legend_label = None , alpha=1 , axis_labels = None ) :
    from bokeh.plotting import figure, output_file, ColumnDataSource
    from bokeh.models   import HoverTool, Range1d, Text
    from bokeh.models   import Arrow, OpenHead, NormalHead, VeeHead, Line
    #
    if 'str' in str(type(color)):
        colors_ = [ color for v in X ]
    else :
        colors_ = color
        
    data = { **{'x' : X , 'y' : Y ,
                'color': colors_ ,
                'alpha': [ alpha for v in X ] } }
    ttips = [   ("index "  , "$index"   ) ,
                ("(x,y) "  , "(@x, @y)" ) ]
    
    if not additional_dictionary is None :
        if 'dict' in str(type(additional_dictionary)):
            data = {**data , **additional_dictionary }
            for key in additional_dictionary.keys() :
                ttips.append( ( str(key) , '@'+str(key) ))
    
    source = ColumnDataSource ( data = data )
    hover = HoverTool ( tooltips = ttips )
    #
    if p is None :
        p = figure ( plot_width=600 , plot_height=600 , 
           tools = [hover,'box_zoom','wheel_zoom','pan','reset','save'],
           title = title )
        
    if legend_label is None :
        p.circle( 'x' , 'y' , size=12, source=source , color='color', alpha='alpha' )
    else :
        p.circle( 'x' , 'y' , size=12, source=source , color='color', alpha='alpha' , legend_label=legend_label )

    p.xaxis.axis_label = axis_labels[ 0] if not axis_labels is None else 'x'
    p.yaxis.axis_label = axis_labels[-1] if not axis_labels is None else 'y'
    p.output_backend = 'webgl'
    
    return( p )


def plotter ( x = np.random.rand(10) , y = np.random.rand(10) , colors = '#ff0000' ,
             legends=None, axis_labels = None, bSave = False, name='scatter.html' ):

    from bokeh.plotting import output_file, show, save
    
    output_file( name )
    outp = lambda x: save if bSave else show

    if list_typecheck([x,y,colors],'list',all) :
        p = bscatter(  [0] , [0] , color = "#ffffff" , alpha=0 )
        for i in range(len(x)) :
            x_ , y_ , color = x[i] , y[i] , colors[i]
            if list_typecheck([legends,axis_labels],'list',all):
                label = legends[i]
                p = bscatter(  x_ , y_ , color = color , p = p , legend_label = label , axis_labels=axis_labels )           
            else :
                p = bscatter(  x_ , y_ , color = color , p = p )
        outp ( p )
    else :
        p = bscatter( x,y, color=colors )
        outp ( p )
    return( p )
