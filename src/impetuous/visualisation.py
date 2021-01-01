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

make_hex_colors = lambda c : '#%02x%02x%02x' % (c[0]%256,c[1]%256,c[2]%256)

def bscatter ( X , Y , additional_dictionary=None , title='' , color='#ff0000' , p=None, legend_label = None , alpha=1 , axis_labels = None ) :
    from bokeh.plotting import figure, output_file, ColumnDataSource
    from bokeh.models   import HoverTool, Range1d, Text
    from bokeh.models   import Arrow, OpenHead, NormalHead, VeeHead, Line
    #
    if 'str' in str(type(color)):
        colors_ = [ color for v in X ]
    else :
        colors_ = color

    if 'list' in str(type(alpha)):
        alphas_ = alpha
    else :
        alphas_ = [ alpha for v in X ]
        
    data = { **{'x' : X , 'y' : Y ,
                'color': colors_ ,
                'alpha': alphas_ } }
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


def plotter ( x = np.random.rand(10) , y = np.random.rand(10) , colors = '#ff0000' , title='',
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
