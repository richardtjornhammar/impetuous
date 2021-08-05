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

from scipy import stats
from scipy.stats import rankdata
#
import matplotlib._color_data as mcd

from bokeh.layouts import row, layout, column
from bokeh.models import Column, CustomJS, Div, ColumnDataSource, HoverTool, Circle, Range1d, DataRange1d, Row
from bokeh.models import MultiSelect
from bokeh.models import Arrow, OpenHead, NormalHead, VeeHead, Line
from bokeh.models.widgets import TextInput, Toggle
from bokeh.plotting import figure, output_file, show, save
#from bokeh.plotting import figure, output_file, show, ColumnDataSource
#from bokeh.models   import HoverTool, Range1d, Text, Row
#
import warnings

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

#
# BELOW CODES WERE ORIGINALLY INVENTED AND AUTHORED BY RICHARD TJÖRNHAMMAR IN EARLY 2017
#
def b2i( blist ):
    ilist = [ idx for idx in range(len(blist)) if blist[idx] ]
    return ( ilist )
#
# LEGACY REFERS TO BOKEH BETA VERSIONS
legacy = False
legacy_dict = {True:0.5,False:0.0}

# MAKE COLOURS COLORS
def return_n_random_color_values(n=1,bRandom=True):
    import matplotlib._color_data as mcd
    l_colors = [name for name in mcd.CSS4_COLORS if "xkcd:" + name in mcd.XKCD_COLORS ]
    colors = list()
    for i in range(n):
        if bRandom:
            colors.append(l_colors[ np.random.randint( 0, len(l_colors)-1 )])
        colors.append(l_colors[ i ])
    return ( colors )

def make_n_colors( num_cases ):
    import matplotlib._color_data as mcd
    l_xkcd_colors = [name for name in mcd.CSS4_COLORS if "xkcd:" + name in mcd.XKCD_COLORS ]
    num_colors = len(l_xkcd_colors) ;
    colors = []
    for ic in range(num_cases) :
        if ic==0 :
            colors.append( 'red' )
        else :
            colors.append( l_xkcd_colors[int(np.floor(ic/(num_cases-1)*(num_colors-1)))] )
    return(colors)

def make_color_dictionary( labels ):
    import matplotlib._color_data as mcd
    l_xkcd_colors = [name for name in mcd.CSS4_COLORS if "xkcd:" + name in mcd.XKCD_COLORS ]
    slabels = list(set(labels))
    num_cases = len(list(set(labels))) ; num_colors = len(l_xkcd_colors) ;
    colormap = dict()
    for ic in range(len(slabels)):
        if ic==0 :
            colormap[slabels[ic]] = 'red'
        else:
            colormap[slabels[ic]] = l_xkcd_colors[int(np.floor(ic/(num_cases-1)*(num_colors-1)))]
    return ( colormap )

def make_color_values( labels , bMonochrome=False ):
    colors = list()
    if bMonochrome:
        for l in labels:
            colors.append('red')
    else:
        colordict = make_color_dictionary( labels=labels )
        for l in labels:
            colors.append(colordict[l])
    return colors


def a_plot( ds, xvar, yvar, tools, xlab, ylab,
            hover_txt=None, multiple_series=None,
            plot_pane_width=400, plot_pane_height = 400,
            color_label=None, legend_label=None, title=None, legend_location=None,
            alpha=0.25, size=7, color=None, xLtype='auto' , yLtype='auto' ):

    x_axis_type = xLtype
    y_axis_type = yLtype

    if 'list' in str(type(alpha)) :
            b_alpha = alpha[0]
            s_alpha = b_alpha
            n_alpha = b_alpha
            if len(alpha) == 2 :
                s_alpha = alpha[1]
            if len(alpha) == 3 :
                n_alpha = alpha[2]
    else :
        b_alpha = alpha

    if hover_txt:
        tools.append(HoverTool(tooltips=hover_txt))
    fig = figure(   plot_width=plot_pane_width, plot_height=plot_pane_height,
                    tools=tools, title=title, x_axis_type=x_axis_type,
                    y_axis_type=y_axis_type )
    if 'log' in x_axis_type :
        fig.xaxis.major_label_orientation = np.pi*0.5

    if multiple_series:
        ch = fig.circle(xvar, yvar, color='col', legend='cat', source=ds, alpha=b_alpha, size=size )
    else :
        if not 'None' in np.str(type(color_label)) and not 'None' in np.str(type(legend_label)) :
            ch = fig.circle(xvar, yvar, source=ds, alpha=b_alpha, size=size, color=color_label, legend=legend_label)
        else:
            if not 'None' in np.str(type(color_label)):
                ch = fig.circle(xvar, yvar, source=ds, alpha=b_alpha, size=size, color=color_label )
            else:
                if not legend_label is None:
                    ch = fig.circle ( xvar, yvar, source=ds, alpha=b_alpha, size=size, legend=legend_label )
                else:
                    ch = fig.circle ( xvar, yvar, source=ds, alpha=b_alpha, size=size )

    if 'list' in str(type(alpha)) and 'list' in str(type(color)) :
        ch.selection_glyph    = Circle( fill_alpha=s_alpha, fill_color=color[0], line_color=color[0] , radius = size*4 )
        ch.nonselection_glyph = Circle( fill_alpha=n_alpha, fill_color=color[1], line_color=None)
    else :
        ch.selection_glyph = Circle(fill_alpha=1.0, fill_color='red', line_color=None, size = size*4 )
        ch.nonselection_glyph = Circle(fill_alpha=0.25, fill_color='blue', line_color = None)

    if not legend_location is None :
        fig.legend.location = legend_location
        fig.legend.orientation = "horizontal"

    fig.xaxis.axis_label = xlab
    fig.yaxis.axis_label = ylab

    return fig


def box_plot (  box_source , categories, not_empty=False, with_line=False, tools=['save'],
                plot_pane_width=400, plot_pane_height=400 , title='',score='', scatter_data_source=None ) :

    p = figure (    tools=tools, background_fill_color="#EFE8E2", title=title, x_range=categories ,
                    plot_width = plot_pane_width, plot_height = plot_pane_height )

    if 'dict' in str(type(box_source)) :
        single_box_dict = dict()
        single_box_dict['cats']  = box_source[ 'cats'  ]
        single_box_dict['us']    = box_source[ score+'us' ]
        single_box_dict['ls']    = box_source[ score+'ls'  ]
        single_box_dict['q2s']   = box_source[ score+'q2s' ]
        single_box_dict['q1s']   = box_source[ score+'q1s' ]
        single_box_dict['q3s']   = box_source[ score+'q3s' ]
        single_box_source = single_box_dict
        if not scatter_data_source is None :
            single_scatter_dict['ncats'] = scatter_data_source['ncats']
            single_scatter_dict['vals'] = scatter_data_source[ score+'vals']
            single_scatter_source = scatter_data_source
    else:
        single_box_source = box_source
        single_scatter_source = scatter_data_source

    p.segment( 'cats', 'us', 'cats', 'q3s', source = single_box_source )
    p.segment( 'cats', 'ls', 'cats', 'q1s', source = single_box_source )
    p.vbar ( 'cats' , 0.7 , 'q2s', 'q3s', source = single_box_source , fill_color='#E08E79', line_color='black' )
    p.vbar ( 'cats' , 0.7 , 'q1s', 'q2s', source = single_box_source , fill_color='#E08E79', line_color='black' )
    p.rect ( 'cats' , 'ls', 0.2, 0.01, source = single_box_source , line_color='black' )
    p.rect ( 'cats' , 'us', 0.2, 0.01, source = single_box_source , line_color='black' )

    if not single_scatter_source is None :
        p.circle( 'ncats' , 'vals' , source = single_scatter_source , size=20, alpha=0.2, color='red' )
    if with_line:
        p.line('cats','q2s', source = single_box_source , line_color='black')
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = "white"
    p.grid.grid_line_width = 2
    p.xaxis.major_label_text_font_size="12pt"
    p.xaxis.major_label_orientation = np.pi/3
    p.xaxis.axis_label = score

    return ( p )

def show_vbar_plot  (   bar_source, tools = ['save'], addHeigth = 70 , init_range=None ,
                        plot_pane_width = 400, plot_pane_height = 400 , title = '',
                        score = '', yaxis_label=''
                    ) :
    p = vbar_plot ( bar_source=bar_source, tools = tools, addHeigth = addHeigth , init_range=init_range ,
            plot_pane_width = plot_pane_width, plot_pane_height = plot_pane_height ,
            title = title, score = score, yaxis_label=yaxis_label )
    output_file("vbarplot.html", title="vbarplot.py example")
    show ( p )


def vbar_plot   (   bar_source, tools = ['save'], addHeigth = 70 , init_range=None ,
                    plot_pane_width = 400, plot_pane_height = 400 ,
                    title = '', score = '', yaxis_label='', r_s=-10.0, r_e=10.0
                ) :
    single_bar_source = bar_source
    if 'dict' in str(type(bar_source)) :
        single_bar_dict = dict()
        single_bar_dict['ids' ] = bar_source[score+'_ids']
        single_bar_dict['y'] = np.array(bar_source[score+'_values'])
        single_bar_source = ColumnDataSource ( dict( x = 0.5+np.array(range(single_bar_dict[score+'_ids'].length()) ),
                                    ids = single_bar_dict['ids'], y = single_bar_dict[score+'_values'] )                                )
        init_range = list( bar_source[score+'_ids'] )

    if 'None' in str(type( init_range )) :
        init_range = ['Null']

    if 'list' in str(type(title)) and len(title)==2 :
        yaxis_label = title[1]
        title       = title[0]

    p = figure (    tools = tools , background_fill_color = "white" ,
                    title = title , x_range = init_range , y_range = Range1d( start=r_s, end=r_e ) , # EYE SORE. NEEDS TO BE SET TO SOMETHING
                    plot_width = plot_pane_width, plot_height = plot_pane_height + addHeigth
                )
    p . vbar (
                x = 'x' , width=0.95 , bottom = 0 , top = 'y' ,
                fill_color = 'firebrick' , line_color = 'black', source = single_bar_source
            )
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = "white"
    p.grid.grid_line_width  = 2
    p.xaxis.major_label_text_font_size = "10pt"
    p.xaxis.major_label_orientation = np.pi/2
    p.xaxis.axis_label = score
    p.yaxis.axis_label = yaxis_label
    return ( p )


def histogram_plot (    nbins=100, color=None, label=None,
                        plot_pane_width=400, plot_pane_height=400 , title='' ) :
    p = figure (
                    background_fill_color='white' , title=title ,
                    plot_width = plot_pane_width , plot_height = plot_pane_height
            )
    p . xgrid.grid_line_color = None
    p . ygrid.grid_line_color = "white"
    p . grid .grid_line_width = 2
    return ( p )


def make_text_search_wb (   name , title='Input name:', variable_name='tval' ,
                            args=None , dep_code = None , triggercode=".change.emit();",
                            bDebug=False ) :
    ecode0 = """
            var ind_str = """+variable_name+""".value ;
            var dat = source.data;
            var pat = pw.data;
            var path_names = pat['pathways'];
            var ind = -1
            for( var i=0 ; i<path_names.length ; i++ ) {
                if( ind_str == path_names[i] ) {
                    ind = i
                }
            }
            dat['x'] = pat['pathway2patient_x'][ind];
            source"""+triggercode+"""
        """
    text_input = TextInput( value=name, title=title )
    toggle = Toggle( label="Search" , button_type="success" )
    
    from bokeh.events import ButtonClick
    from bokeh.models import Button, CustomJS
    button = Button()

    new_code = []
    div = Div(width=500)
    if not dep_code is None :
        for line in dep_code.split('\n') :
            add_code = None
            if 'cb_obj' in line:
                line = """        var ind_str = """ + variable_name + """.value ;"""
            if '= pat[' in line and '[ind]' in line:
                add_code = [
                    '        var stats = left.data;',
                    '        var confidence = stats[\'y\'];',
                    '        var path_names = pat[\'pathways\'];',
                    '        var path_descr = pat[\'descriptions\'];',
                    '        if ( path_descr == undefined ) {',
                    '            path_descr=[];',
                    '            for ( var i=0 ; i<path_names.length ; i++ ) {',
                    '                path_descr.push(-1);',
                    '            }',
                    '        }',
                    '        var ind = -1;',
                    '        var v_ind = [];',
                    '        var resArray = []',
                    '        for ( var i=0 ; i<path_names.length ; i++ ) {',
                    '            var lpn = String(path_names[i]).toLowerCase();',
                    '            var ldn = String(path_descr[i]).toLowerCase();',
                    '            var lsistr = ind_str.toLowerCase();',
                    '            if ( lpn.search(lsistr)!=-1 || ( path_descr[i]!=-1 && ldn.search(lsistr)!=-1 ) ) {',
                    '                resArray.push([ confidence[i],"[ "+path_names[i]+" ] : < "+confidence[i]+" > : [ "+path_descr[i]+" ] <br>" , i]); ',
                    '            }',
                    '        }',
                    '        div.text = "<b>" + "Group ID : < Significance q (left,y-axis) > : Group description" + "</b><br>" ; var first = 1;',
                    '        resArray.sort( function(a,b){ return a[0]-b[0] } );',
                    '        for ( var i=0 ; i<resArray.length ; i++ ) {',
                    '                if (first==1) { div.text += "<b>" ; }',
                    '                div.text += resArray[i][1]; ',
                    '                if (first==1) { div.text+=" </b> " ; first = 0; }',
                    '        }',
                    '        ind = resArray[0][2];',
                    '        var selection = left.selected.indices',
                    '        selection[0] = resArray[0][2];',
                    '        left' + triggercode,
                ]
            if not add_code is None :
                [ new_code .append( ac ) for ac in add_code ]
            new_code .append( line )
    if not args is None :
        arguments = { **args, **{variable_name:text_input} }
    if dep_code is None:
        ecode = ecode0
    else :
        ecode = '\n'.join(new_code)
    #button.js_on_event(ButtonClick , CustomJS(code='console.warn("JS:Click")') )
    button.js_on_event(ButtonClick,  CustomJS( args={**arguments,**dict(div=div)}, code=ecode ) )
    if bDebug :
        print ( dep_code )
        print (  '***'   )
        print (  ecode0  )
        print (  '***'   )
        print ( '\n'.join(new_code) )
    wbox = Column ( text_input , button )
    return ( column(wbox,div) )



def scatter2boxplot(    pathways, x1, y1, variables, patients, pathway2patient_x, patient_y, patient_type,
                        axis_labels = None, out_file_name = "callback.html", category_override = 'Time',
                        dynamic_labelx = None, dynamic_labely = None, pathway2patient_y = None ,title_l=None, title_r=None,
                        patient_pos = None, category_pos = None, categories = None, xLtype='auto', yLtype='auto',
                        chart_dicts = None, bShow = False, chart_titles = None , with_input_search = True ) :
    #
    triggercode=".change.emit();"
    if legacy :
        triggercode=".trigger('change');"
    if not 'None' in str(type(categories)):
        spacer = np.max([ len(ck) for ck in [categories[ck] for ck in list(categories.keys())] ])*10
    else:
        spacer = 0
    #
    patient_names = [ p[patient_pos] for p in patients ]
    if not 'None' in str(type(category_pos)):
        patient_times = [ categories[p[category_pos]] for p in patients ]
    else:
        patient_times = [ p[category_pos] for p in patients ]
    #
    cat2num = { v:int(k) for k,v in categories.items() }
    num_cats = [ int(cat2num[cval])+0.5 for cval in patient_times ]
    reg_cats = patient_times
    #
    assert( not np.isnan(x1).any() )
    assert( not np.isnan(y1).any() )
    #
    output_file(out_file_name)
    if axis_labels == None:
        axis_labels = {'x1': 'x-label','y1': 'y-label','x2': 'x-label','y2': 'y-label',0: ('healthy',"blue"),1: ('sick','red')}
    src_dict = dict( x=x1, y=y1, pathways=pathways )
    src_dict .update(variables)
    #
    hover_txt = [("index", "$index"),
        (axis_labels['x1'], "@x"),
        (axis_labels['y1'], "@y"),
        ("pathway", "@pathways")]
    for key in variables.keys():
        hover_txt.append((str(key),"@"+str(key)))
    left_pane_data_source = ColumnDataSource(data=src_dict)
    tw = 0
    if not 'None' in str(type(title_l)):
        tw = len(title_l)-40
    left_figure = a_plot( left_pane_data_source,'x','y',['tap','box_zoom','wheel_zoom','pan','reset','save'], axis_labels['x1'], axis_labels['y1'],
                          hover_txt, False, title=title_l, alpha=[0.75,1.0,0.25], color=['red','blue'] ,
                          plot_pane_width=tw*(tw>0)*3+400, plot_pane_height= spacer+400,
                          xLtype=xLtype , yLtype=yLtype )
    #
    patient_data = dict(x = [y for y in patient_y], y = patient_y,
                        patients = patients,
                        cat = [axis_labels[pat_type][0] for pat_type in patient_type],
                        col = [axis_labels[pat_type][1] for pat_type in patient_type] )
    #
    path_data = dict( pathways=pathways , pathway2patient_x=pathway2patient_x, numcats=num_cats , cats = reg_cats)
    #
    if not 'None' in str(type(pathway2patient_y)) :
        path_data['pathway2patient_y'] = pathway2patient_y
    bHaveDesc = False
    if 'list' in str( type(title_r) ) :
        if len(title_r) == len(pathways) :
            bHaveDesc = True
            path_data['descriptions'] = title_r
            title_r = 'Description'
        else :
            title_r = title_r[0]
    #
    pathway_data_source = ColumnDataSource( path_data )
    #
    box_orig_data = dict()
    box_orig_data['vals'] = [] ; box_orig_data['ncats'] = [] ;
    additional_data_source = ColumnDataSource ( box_orig_data )

    box_patient_data = dict()
    box_patient_data['cats'] = [] ; box_patient_data['us']  = [] ; box_patient_data['ls']  = []
    box_patient_data['q1s']  = [] ; box_patient_data['q2s'] = [] ; box_patient_data['q3s'] = []
    right_pane_data_source = ColumnDataSource ( box_patient_data )

    right_figure = box_plot (   right_pane_data_source ,
                                categories = [categories[ck] for ck in list(categories.keys())] , not_empty = False,
                                with_line = False , title = title_r, score=pathways[0] , plot_pane_height = spacer+400 ,
                                tools=['tap','box_zoom','wheel_zoom','pan','reset','save'], scatter_data_source = additional_data_source
                            )

    add_right_figure = None
    vb_tail_scripts = list(); vb_charts = list() ; vb_chart_dicts = list()
    if 'list' in str( type( chart_dicts ) ):
        for cnum in range(len( chart_dicts )):
            chart_dict = chart_dicts[cnum]
            if 'dict' in str(type( chart_dict )) :
                pathway_gene_dict = chart_dict
                NUMBER = 2+cnum ;

                crop = set([key for key in pathway_gene_dict.keys() if ( ('_ids' in key) or ('_values' in key) )])
                an_id = list(crop)[0].split('_')[0]

                adict = pathway_gene_dict
                ndict = {k:adict[k] for k in crop if k in adict}
                gene_bar_ds = ColumnDataSource( data = ndict )
                add_right_data_source = ColumnDataSource( dict(  x=ndict[an_id+'_ids'], y=ndict[an_id+'_values'], ids=ndict[an_id+'_ids'] ) )
                ctitle = None
                if not 'None' in chart_titles:
                    if len(chart_titles) > cnum :
                        if not 'None' in chart_titles[ cnum ]:
                            ctitle = chart_titles[ cnum ]
                add_right_figure = vbar_plot ( bar_source = add_right_data_source, init_range=list(ndict[an_id+'_ids']),
                                        title=ctitle, tools=['box_zoom','tap','wheel_zoom','pan','reset','save'] )
                vb_add_dict = dict ( {'source'+str(NUMBER):add_right_data_source, 'fig'+str(NUMBER):add_right_figure, 'genes'+str(NUMBER):gene_bar_ds} )
                vb_charts .append(add_right_figure)
                vb_chart_dicts .append(vb_add_dict)

                vb_add_script = """ """
                if 'None' in str(type(ctitle)):
                    vb_tail_script = """fig"""+str(NUMBER)+"""['title'].text = path_name ;
                    """
                else :
                    vb_tail_script = """ """

                vb_tail_script += """
                ymax = 0 ; ymin = 0 ;
                for( var i=0 ; i<genes"""+str(NUMBER)+""".data[ path_name+'_values' ].length ; i++ ) {
                    if( genes"""+str(NUMBER)+""".data[ path_name+'_values' ][i]>ymax ) {
                        ymax = genes"""+str(NUMBER)+""".data[ path_name+'_values' ][i]
                    }
                    if(genes"""+str(NUMBER)+""".data[ path_name+'_values' ][i]<ymin ) {
                        ymin = genes"""+str(NUMBER)+""".data[ path_name+'_values' ][i]
                    }
                }
                fig"""+str(NUMBER)+"""['y_range']['start'] = ymin - 1.0
                fig"""+str(NUMBER)+"""['y_range']['end'] = ymax + 1.0
                """
                vb_tail_script += """
                fig"""+str(NUMBER)+"""['below'][0]['axis_label'] = path_name ;
                fig"""+str(NUMBER)+"""['x_range'].start = """ + str(legacy_dict[legacy]) + """;
                fig"""+str(NUMBER)+"""['x_range'].end = genes"""+str(NUMBER)+""".data[ path_name+'_ids' ].length + """ + str(legacy_dict[legacy]) + """;
                fig"""+str(NUMBER)+"""['x_range'].factors = genes"""+str(NUMBER)+""".data[ path_name+'_ids' ]
                fig"""+str(NUMBER)+triggercode+"""
                var inf"""+str(NUMBER)+""" = source"""+str(NUMBER)+""".data;
                inf"""+str(NUMBER)+"""['x'] = new Array( genes"""+str(NUMBER)+""".data[ path_name+'_ids' ].length);
                for(var i=0;i<inf"""+str(NUMBER)+"""['x'].length;i++) {
                    inf"""+str(NUMBER)+"""['x'][i] = i + """ + str(0.5+legacy_dict[legacy]) + """
                }
                inf"""+str(NUMBER)+"""['ids'] = genes"""+str(NUMBER)+""".data[ path_name+'_ids'];
                inf"""+str(NUMBER)+"""['y'] = genes"""+str(NUMBER)+""".data[ path_name+'_values' ];
                source"""+str(NUMBER)+triggercode+"""
                """
                vb_tail_scripts.append(vb_tail_script)
    else :
        vb_add_script = '' ; vb_add_dict = dict() ; vb_tail_script = '';
        vb_tail_scripts = list(vb_tail_script)
        vb_charts = None
        vb_chart_dicts = list(vb_add_dict)

    add_functions = """
function onlyUnique(value, index, self) {
    return self.indexOf(value) === index;
}
function median(values){
    var mvals = values.slice()
    mvals.sort( function(a,b) {return a - b;} );
    var half = Math.floor(mvals.length*0.5);
    var med;
    if ( mvals.length % 2 )
        med = mvals[half];
    else
        med = (mvals[half-1] + mvals[half]) * 0.5;
    return( med );
}
function quartileBounds(values,groups) {
    var uvals = groups.filter(onlyUnique)
    var dict = {}
    var LS=Array() ; var Q1=Array() ; var Q2=Array() ;
    var Q3=Array() ; var US=Array() ; var G =Array() ;
    for ( var i=0 ; i<uvals.length ; i++ ) {
        var gvalues = Array();
        for ( var j=0 ; j < values.length ; j++ ) {
            if( groups[j]==uvals[i] ) {
                gvalues.push(values[j]);
            }
        }
        var medi = median(gvalues);
        var lowerHalf = gvalues.filter(function(f){ return f < medi });
        var upperHalf = gvalues.filter(function(f){ return f >= medi });
        var q1s  = median(lowerHalf);
        var q2s  = medi;
        var q3s  = median(upperHalf);
        var iqr  =  q3s - q1s ;
        var iqru = (q3s - q2s) ;
        var iqrd = (q2s - q1s) ;
        var lowq = q1s - 1.5*iqrd ;
        var uppq = q3s + 1.5*iqru ;
        US.push(uppq) ; LS.push(lowq) ; G.push(uvals[i])
        Q1.push(q1s) ; Q2.push(q2s) ; Q3.push(q3s)
    }
    return ( Array( LS.slice() , Q1.slice() , Q2.slice() , Q3.slice() , US.slice() , G.slice() ) ) ;
}
    """
    # var ind  = cb_obj.selected['1d'].indices[0];
    bokeh_script = add_functions + """
        var ind  = cb_obj.indices[0];
        var dat  = source.data;
        var dats = sources.data;
        var pat  = pw.data;
        var path_name = pat['pathways'][ind];
        var nums   = pat['pathway2patient_x'][ind];
        var groups = pat['numcats'];
        var qs = quartileBounds(nums,groups)

        dat[ 'ls' ] = qs[0];
        dat['q1s' ] = qs[1];
        dat['q2s' ] = qs[2];
        dat['q3s' ] = qs[3];
        dat[ 'us' ] = qs[4];
        dat['cats'] = qs[5];

        dats['ncats'] = pat['numcats'];
        dats['vals' ] = pat['pathway2patient_x'][ind];
    """

    if bHaveDesc :
        bokeh_script += """
        var path_desc = pat['descriptions'][ind];
        """
    bokeh_script += """
        source"""+triggercode+"""
        sources"""+triggercode+"""
    """
    if bHaveDesc:
        bokeh_script += """
        fig['title'].text = path_desc;\n
        fig['title']"""+triggercode+"""\n
    """
    if not 'None' in np.str(type(dynamic_labely)) :
            bokeh_script +="fig['left'][0]['axis_label'] = '"+dynamic_labely[0]+"' + path_name + '"+dynamic_labely[1]+"';\n"
            bokeh_script +="fig['left'][0]" + triggercode+"\n"
    if not 'None' in np.str(type(dynamic_labelx)) :
            bokeh_script +="fig['below'][0]['axis_label'] = '"+dynamic_labelx[0]+"' + path_name + '"+dynamic_labelx[1]+"';\n"
            bokeh_script +="fig['below'][0]" + triggercode+"\n"
    else :
        bokeh_script +="fig['left' ][0]['axis_label'] = path_name ;\n"
        bokeh_script +="fig['below'][0]['axis_label'] = 'Category' ;\n"
        bokeh_script +="fig['left' ][0]" + triggercode+"\n"
        bokeh_script +="fig['below'][0]" + triggercode+"\n"

    for vb_tail_script in vb_tail_scripts:
        bokeh_script += vb_tail_script
    arg_dict = dict(    source=right_pane_data_source, fig = right_figure, pw = pathway_data_source ,
                        sources=additional_data_source )

    arguments = arg_dict
    for vd_add_dict in vb_chart_dicts:
        arguments.update(vd_add_dict)
    print([item for item in arguments.keys()],bokeh_script)
    left_pane_data_source.selected.js_on_change('indices',  CustomJS( args=arguments, code=bokeh_script) )

    if not vb_charts is None :
        all_plots = [ left_figure, right_figure ]
        for chart in vb_charts:
            all_plots.append( chart )
        lo = gridplot( all_plots, ncols=2, inner_plot_width=400, inner_plot_height=400, merge_tools=False )
    else :
        lo = row( left_figure, right_figure )
    #
    if with_input_search :
        arguments.update(dict(left=left_pane_data_source))
        text_wb = make_text_search_wb ( 'Name of pathway', variable_name='uinp', args=arguments, dep_code=bokeh_script )
        clo = row(text_wb,lo)
    else :
        clo = lo
    #
    if bShow :
        show ( clo )
    else :
        save ( clo )

import re
def generate_dict( df , dict_type='D', exclude_str=None, super_safe=False,
                   graph_labels=['x','y'], which_xy=None ):

    if len(graph_labels)>2:
        color_label = graph_labels[2]
    else:
        color_label = None
    if len(graph_labels)>3:
        legend_label = graph_labels[3]
    else:
        legend_label = None

    if 'None' in np.str(type(which_xy)):
        want_x = df.index[0]; want_y = df.index[1]
    else:
        want_x = which_xy[0] ; want_y = which_xy[1]
    rv_dict = dict( )

    if dict_type=='S' : # SAMPLE
        exclude = [ name for name in df.columns.names ]
        if color_label:
            exclude.append(color_label)
        if legend_label:
            exclude.append(legend_label)

        orig_idx = df.index.values
        ydat=[]; xdat=[]; zdat=[]; coldat=[]; legdat=[];
        for ic in range(len(orig_idx)) :
            if orig_idx[ic] in set(exclude):
                continue
            else:
                if exclude_str:
                    if exclude_str in orig_idx[ic] :
                        continue
            [ ydat.append( yval ) for yval in df.loc[ orig_idx[ic] ].values ]
            [ xdat.append( xval ) for xval in np.ones(len(df.loc[orig_idx[ic]].values))*ic ]
            if color_label:
                [ coldat.append( cval ) for cval in df.loc[ color_label ].values ]
            if legend_label:
                [ legdat.append( cval ) for cval in df.loc[ legend_label ].values ]
            ncnts = len( df.loc[ orig_idx[ic] ].values)
            for icc in range(ncnts):
                zdat . append( orig_idx[ic] )
        rv_dict[graph_labels[0]]=zdat; rv_dict[graph_labels[1]]=ydat
        if color_label:
            rv_dict[ color_label]=coldat
        if legend_label:
            rv_dict[legend_label]=coldat
    else :
        bCols = np.ones(len(df.columns))>0
        rv_dict[graph_labels[0]]=df.loc[want_x, bCols ]
        rv_dict[graph_labels[1]]=df.loc[want_y, bCols ]
        if color_label:
            rv_dict[ color_label] = df.loc[ color_label, bCols ]
        if legend_label:
            rv_dict[legend_label] = df.loc[legend_label, bCols ]
        for idx in df.index:
            if idx in set(graph_labels) :
                continue
            if exclude_str :
                if (exclude_str in idx):
                    continue
            safe_hover_idx = ((idx.replace(',','_')).replace('-','_')).replace(' ','_')
            if super_safe:
                safe_hover_idx = "".join(re.findall("[a-zA-Z0-9]+",idx))
            rv_dict[ safe_hover_idx.replace('void','') ] = df.loc[ idx ]
        rv_dict['name'] = df.columns.values
    return( rv_dict )

from bokeh.layouts import gridplot

def add_warning( script,dump_str='ind + fine_name' ):
    alert_txt = """window.alert(""" + dump_str + """);"""
    script += alert_txt + '\n'
    return script

def safe_labels( axis_labels, i=0, graph_labels=['x','y']):
    x_label=graph_labels[0] ; y_label = graph_labels[1]
    if not 'None' in np.str(type(axis_labels)):
        if len(axis_labels)>i:
            if len(axis_labels[i])==2:
                x_label = axis_labels[i][0]
                y_label = axis_labels[i][1]
    return x_label,y_label


def base_bokeh_tri_plot ( bokeh_data, fname = None , gwidth = 400 , gheight=None,
                    bDebug = False, lb_types=None, hover_scripts=None,
                    graph_labels = [ 'x','y','z' ], axis_labels = None ,
                    color_label = None, legend_label = None, show_legends=None ) :
    if fname:
        output_file(fname)
    if 'None' in np . str( type( gheight ) ):
        gheight = gwidth
    if 'None' in np.str(type(show_legends)):
        show_legends=[True,True,True]
    vlegend_label=list()
    if not 'None' in np.str(type(legend_label)):
        for s in show_legends:
            if s:
                vlegend_label.append(legend_label)
            else:
                vlegend_label.append(None)
    else:
        for s in show_labels:
            vlegend_label.append(None)
    nmh = 3*(np.max([ len(col) for col in bokeh_data[4][3] ]) + 1)
    if not nmh:
        nmh=0
    d = 0
    if lb_types:
        if lb_types[0]:
            if lb_types[0]=='S':
                d=nmh

    bSavePlot = not ( 'None' in np.str(type(fname)) ) ; fname =' '
    bokeh_lab = graph_labels

    cjs_args = bokeh_data[0][2]
    toolbox = [ ['tap','box_zoom','wheel_zoom','pan','reset','save'],
                ['box_zoom','wheel_zoom','pan','reset','save'],
                ['tap','box_zoom','wheel_zoom','pan','reset','save'] ]
    tb = []
    for it in range(len(toolbox)):
        tool = toolbox[it]
        if not ( 'None' in np.str(type(hover_scripts)) ):
            if it < len(hover_scripts) :
                hover = HoverTool( tooltips = hover_scripts[it] ) #, callback = hover_callback_L, show_arrow=False )
                tool.append( hover )
        tb.append(tool)
    toolbox = tb

    x_label, y_label = safe_labels(axis_labels,i=0)
    pane_bottom = a_plot(   ds=bokeh_data[0][0],xvar=bokeh_lab[0],yvar=bokeh_lab[1],tools=toolbox[0],
                            xlab=x_label,ylab=y_label,plot_pane_width=gwidth, plot_pane_height = gheight,
                            color_label=color_label, legend_label=vlegend_label[0]) #legend_label )

    x_label, y_label = safe_labels(axis_labels,i=1)
    pane_right = a_plot(    ds=bokeh_data[1][0],xvar=bokeh_lab[0],yvar=bokeh_lab[1],tools=toolbox[1],
                            xlab=x_label,ylab=y_label,plot_pane_width=gwidth, plot_pane_height = gheight + d,
                            color_label=color_label,legend_label=vlegend_label[1] )
    cjs_args['fig'] = pane_right

    x_label, y_label = safe_labels(axis_labels,i=2)
    pane_left = a_plot(     ds=bokeh_data[2][0],xvar=bokeh_lab[0],yvar=bokeh_lab[1],tools=toolbox[2],
                            xlab=x_label,ylab=y_label,plot_pane_width=gwidth, plot_pane_height = gheight + d,
                            color_label=color_label,legend_label=vlegend_label[2] )
    if lb_types:
        if lb_types[0]:
            if lb_types[0]=='S':
                bCategorical = True
                pane_left.xaxis.major_label_orientation = np.pi*0.5
                pane_left.x_range = left_data[bokeh_lab[0]]

    grid = gridplot( [[pane_left, pane_right], [None,  pane_bottom]] , merge_tools=False)
    #
    # THESE ARE ADDED HERE BECAUSE OF THE FIGURE ADDITION TO CJS_ARGS
    bokeh_data[2][0].callback = CustomJS ( args=cjs_args , code = bokeh_data[2][3] ) # LEFT
    bokeh_data[0][0].callback = CustomJS ( args=cjs_args , code = bokeh_data[0][3] ) # BOTTOM
    show( grid )
    return 0

def make_hover_information( sample_items , variable_names=None , NameLast=True , name_var = 'z' ) :
    if 'None' in np.str(type(variable_names)) :
        variable_names = sample_items
    rcode0 = """\n\t<div style="width:250px; heigth=50px;">\t"""
    rcoden = """\n\t\t<div>\n\t\t\t<span style="font-size: 12px; font-weight: bold;">@"""+name_var+"""</span>\n\t\t</div>\n"""
    if not NameLast:
        rcode0 += rcoden
    for isi in range ( len ( sample_items ) ) :
        if variable_names[isi]==name_var:
            continue
        rcodes= """\n\t\t<div>\n\t\t\t<span style="font-size: 12px; color: #696;"> """ + variable_names[isi] + """: @""" + sample_items[isi] + """ </span>\n\t\t</div>"""
        rcode0 += rcodes
    if NameLast:
        rcode0 += rcoden
    rcode0+="""
    </div>"""
    return( rcode0 )

def dict_to_hover( data_dictionary , duplicate_keys = None, spec_keys=None ):
    use_keys = list(data_dictionary.keys())
    key_names= None
    if not 'None' in np.str(type(duplicate_keys)):
            use_keys = list( data_dictionary.keys()-set(duplicate_keys) )
            if not 'None' in np.str(type(spec_keys)):
                iSel = b2i( [(np.sum([( sk in uk ) for sk in spec_keys])>0) for uk in use_keys ] )
                use_keys = [ use_keys[i] for i in iSel ] ; key_names=list()
                for uk in use_keys :
                    key_names . append( ''.join([ sk  for sk in spec_keys if (sk in uk) ]) )
    return ( make_hover_information( use_keys, variable_names=key_names, NameLast=True, name_var='name' ) )

def safe_wxy( which_xy , i=0 ):
    wxy=None
    if which_xy:
        if which_xy[i]:
            if len(which_xy[i])==2:
                wxy=which_xy[i]
    return wxy

def write_script(   graph_labels=['x','y'] , source_names=['s2','s4'], link=['s2','s4'], axis=0 ,
                    dict_type='S' , NameLabel=None, dynamic_label=None ):
    axtag = dict() ; axtag[0] = 'below' ; axtag[1] = 'left'
    # .selected['1d'].indices[0] ;
    script_0 = """
    var ind = cb_obj.indices[0];
    var dat = cb_obj.data;
    """
    if dict_type=='S' :
        script_0 += """var fine_name = dat['"""+graph_labels[int(not axis)]+"""'][ind];\n"""
    else:
        if NameLabel:
            script_0 += """var fine_name = dat['"""+NameLabel+"""'][ind];\n"""
        else:
            script_0 += """var fine_name = dat['name'][ind];\n"""
    ddict = dict()
    for i in range(len(source_names)):
        addstr = '    var d'+np.str(i+1)+' = ' +source_names[i]+ '.data;\n'
        script_0 += addstr
        ddict[source_names[i]] = np.str(i+1)
    script_0 += '    '
    if dict_type=='S' :
        script_0 += "d"+ddict[link[0]]+"['"+graph_labels[axis]+"'] = d"+ddict[link[1]]+"[fine_name]\n"
    else:
        script_0 += "d"+ddict[link[0]]+"['"+graph_labels[axis]+"'] = d"+ddict[link[1]]+"['"+graph_labels[axis]+"'][ind]\n"
    script_0 += '    '+source_names[0]+""".change.emit();"""
    sc0 = """\n    fig['"""+axtag[axis]+"""'][0]['axis_label'] = fine_name;"""
    if not 'None' in np.str(type(dynamic_label)) :
        if len(dynamic_label)==2:
            sc0 += """\n    fig['"""+axtag[axis]+"""'][0]['axis_label'] = '"""+dynamic_label[0]+"""' + fine_name + '"""+dynamic_label[1]+"""';"""
    script_0 += sc0
    script_0 += """\n    fig['"""+axtag[axis]+"""'][0].change.emit();
    """
    return script_0

def safe_dict_type(lb_types,i=0):
    d_type='D'
    if lb_types:
        if lb_types[i]:
            if lb_types[i][0]:
                d_type=lb_types[i][0]
    return d_type


def make_bokeh_input(   pd_bottom, pd_left, pd_meta, pd_bottom2right, main_meta_label='Weigth', which_xy=None, dynamic_label=None,
                        color_label=None, legend_label=None, lb_types = None , bDebug=False, pathway_name_label='name',
                        spec_keys = ['PatientLabel','Age', 'Weight', 'Length', 'BMI', 'Waist', 'Hip'] ) :

    graph_labels = [ 'x','y' ]
    if color_label:
        graph_labels.append(color_label)
    if legend_label:
        graph_labels.append(legend_label)
    source_names = [ 's2','s3','s4' ]
    #
    # BOTTOM PLOT
    bottom_type = safe_dict_type( lb_types,1 )
    wxy = safe_wxy(which_xy,i=0)
    bottom_dict = generate_dict( df=pd_bottom , exclude_str='_', dict_type=bottom_type, graph_labels=graph_labels, which_xy=wxy )

    meta_dict = dict(); M=[]
    for i in range(len(pd_left.columns)):
        if pd_left.columns.values[i] in pd_meta.index.values:
            M . append( pd_meta.loc[pd_left.columns[i]].values.tolist() )
    meta_dict[graph_labels[1]] = M

    bottom2right_dict = dict(); B=[]
    for i in range(len(pd_bottom.columns)):
        if pd_bottom.columns.values[i] in pd_bottom2right.index.values:
            B . append( pd_bottom2right.loc[pd_bottom.columns[i]].values.tolist() )
    bottom2right_dict[graph_labels[0]] = B

    wxy = safe_wxy( which_xy,i=1 )
    dynamic_dict = generate_dict( df=pd_meta, graph_labels=graph_labels, which_xy=wxy , super_safe=True )

    left_type = safe_dict_type( lb_types,0 )
    wxy = safe_wxy( which_xy,i=2 )
    left_dict = generate_dict( df=pd_left , dict_type=left_type, graph_labels=graph_labels, which_xy=wxy )

    bottom_script = write_script( graph_labels=graph_labels , source_names=source_names, link=['s2','s3'], axis=0,
            NameLabel=pathway_name_label, dict_type=bottom_type, dynamic_label=dynamic_label )
    left_script = write_script( graph_labels=graph_labels , source_names=['s2','s4'], link=['s2','s4'], axis=1, NameLabel='name', dict_type=left_type )

    hover_scripts = [ dict_to_hover( bottom_dict , graph_labels ) ]
    hover_scripts.append( dict_to_hover ( dynamic_dict , graph_labels ,
                            spec_keys=spec_keys )
                        )
    if left_type == 'D' :
        hover_scripts.append( dict_to_hover( left_dict , graph_labels ) )

    bottom_variables = ColumnDataSource ( data = bottom_dict )
    left_variables = ColumnDataSource( data = left_dict )

    dynamic_variables = ColumnDataSource( data = dynamic_dict )
    bottom2right_variables = ColumnDataSource ( data = bottom2right_dict )
    meta_variables = ColumnDataSource( data = meta_dict )

    cjs_args = dict() ;
    cjs_args[ source_names[ 0 ] ] = dynamic_variables
    cjs_args[ source_names[ 1 ] ] = bottom2right_variables
    cjs_args[ source_names[ 2 ] ] = meta_variables

    if bDebug:
        bottom_script = add_warning( bottom_script )
        left_script = add_warning( left_script )

    bokeh_data=list()
    bokeh_data.append( [bottom_variables, 'sX', cjs_args , bottom_script ] ) # B
    bokeh_data.append( [dynamic_variables, source_names[0],'dynamic'] )      # R
    bokeh_data.append( [left_variables, 'sY', 'left' , left_script ] )       # L
    bokeh_data.append( [bottom2right_variables, source_names[1],'bottom2right'])
    bokeh_data.append( [meta_variables, source_names[2],'meta', list(meta_dict.keys())] )

    return graph_labels, hover_scripts, bokeh_data

def bokeh_tri_plot ( pd_bottom, pd_left, pd_meta, pd_bottom2right, meta_label ,
    left_bottom_data_types=['DESCRIPTIVE','DESCRIPTIVE'], which_xy=None ,
    pathway_name_label='name', axis_labels=None , color_label=None,
    legend_label=None, file_name=None, show_legends=None, dynamic_label=None ) :
# DESIGN CHOICES: WE WANT TO GENERATE THREE PLOTS WHERE THE TOPRIGHT CONTAINS THE CONNECTION
# BETWEEN THE DATA TO THE LEFT (COARSE DATA)
# AND THE DATA ON THE BOTTOM (FINE DATA)
# THE LEFT AND BOTTOM DATA FRAMES SHOULD COME WITH THE OPION OF DEPICTING THE SAMPLE
# SPACE WITH EITHER SAMPLE LABELS ON THE X AXIS
# OR DESCRIPTIVE STATISTICS OF THE SAME DATA (DEFAULT)
    [ graph_labels, hover_scripts, bokeh_data ] = make_bokeh_input (
            pd_bottom, pd_left, pd_meta, pd_bottom2right, main_meta_label = meta_label, which_xy=which_xy, dynamic_label=dynamic_label,
            color_label = color_label, legend_label = legend_label, lb_types = left_bottom_data_types , pathway_name_label=pathway_name_label )

    base_bokeh_tri_plot( bokeh_data=bokeh_data, fname = file_name ,
                    lb_types = left_bottom_data_types, hover_scripts = hover_scripts ,
                    axis_labels = axis_labels, color_label = color_label,
                    legend_label = legend_label, show_legends=show_legends )

def merge_list_to_rows(pt1,pt2):
    merged = pt1
    [ [ merged[ivec].append( val ) for val in pt2[ivec] ] for ivec in range(len(merged)) ]
    return merged





    
def invert_dictlist(effects):
    #
    # CREATE A SAMPLE EFFECT LOOKUP
    s_effects = {}
    lkdc = [e for e in effects.values()][0]
    for k in lkdc.keys() :
        for v in list(lkdc[k]) :
            s_effects[v] = k
    return(s_effects)

def parse_label_order_entry ( data_df ,
                              label = None ,
                              order = None ):
    #
    if not label is None :
        sv = set(data_df.loc[label].values)
        order_ = list( sv )
        if not order is None :
            if len(set(order)-sv)==0 :
                print (order)
                order_ = order
        order = order_
    else :
        if order is None :
            return ( None,None )
        else :
            look_for_set = set(order)
            label = data_df.index.values[[ len( set('.'.join([str(v) for v in vs]).split('.')) - look_for_set ) == 0 for vs in data_df.values ]][0]
    return( label , order )


def create_paired_figure_p ( data_df , feature_name = None ,
        sample_name_label = None , pairing_label = None, title_prefix = '',
        case_label = None , hue_label = None , effect_label = None,
        pairing_order = None , effect_order = None , yaxis_label = '' ,
        case_order = None , case_mapping = None , hue_order = None ,
        hue_colors = None , plot_height = 600 , bVerbose = False , pReturn=False ) :
    #
    if feature_name is None:
        feature_name = data_df.index.values[0]

    colors_ = [ '#543005','#8c510a','#bf812d','#dfc27d',
                '#f6e8c3','#f5f5f5','#c7eae5','#80cdc1',
                '#35978f','#01665e','#003c30'] # COLORBLIND SAFE DIVERGENT
    #
    # FIGURE STYLES
    plot_dimensions = [ None , plot_height ]
    icon_size , icon_alpha = 10 , 0.6
    dp_ = 100
    global_spot_size  = 14
    global_spot_alpha = 0.33
    major_label_text_font_size  = [ '20pt' , '20pt' ]
    minor_label_text_font_size  = [ '18pt' , '18pt' ]
    major_box_label_orientation = [   0.0  ,  0.0   ]
    textfont, textsize, textstyle, textangle= 'Arial','18pt','bold',0.0
    #
    verbose = False
    if verbose :
        print ( 'TESTING THE NEW VISUALISATION' )
    if sample_name_label is None:
        print ( 'ERROR' )
        return(0)
    #
    # EXCHANGE FOR CHECKS
    case_label , case_order = parse_label_order_entry ( data_df ,
        label = case_label , order = case_order )
    effect_label , effect_order = parse_label_order_entry ( data_df ,
        label = effect_label , order = effect_order )
    pairing_label , pairing_order = parse_label_order_entry ( data_df ,
        label = pairing_label , order = pairing_order )
    hue_label , hue_order = parse_label_order_entry ( data_df ,
        label = hue_label , order = hue_order )
    #
    if verbose :
        print ( hue_label     , hue_order     ) # DIFFERENT COLORS FOR DIFFERENT HUES
        print ( effect_label  , effect_order  ) # THE CATEGORIES THAT ARE INTERESTING
        print ( pairing_label , pairing_order ) # NUMBER OF STEPS IN EACH CASE
        print ( case_label    , case_order    ) # DIFFERENT SYMBOLS FOR DIFFERENT CASES
    #
    all_labels = [ hue_label , effect_label ,
                   pairing_label , case_label ,
                   sample_name_label ]
    label_set  = set ( all_labels )
    #
    # LEXICAL SORT THEN SORTS SUBLEVES ACCORDING TO THE PAIRING ORDER
    # SO WE CREATE A NUMERICAL MAPPING SO THAT THE SUBENTRIES GET SORTED
    # CORRECTLY
    #
    pairing_order_dict = { pairing_order[i]:i for i in range(len(pairing_order)) }
    tuple_names = [ (s,t) for s,t in zip(*data_df.loc[ [sample_name_label,pairing_label] , : ].values)]
    tuples      = [ ( tup[0],pairing_order_dict[tup[1]] ) for tup in tuple_names ]
    multiindex  = pd.MultiIndex.from_tuples(tuples, names = [ sample_name_label, pairing_label ] )
    data_df  .columns = pd.MultiIndex.from_tuples(tuple_names, names = [ sample_name_label, pairing_label ] )
    data_df  = data_df.iloc[ :,multiindex.sortlevel(sample_name_label)[-1] ]
    #
    # NOW DATA IS SORTED. COLLECT DATA AND STYLE DATA FRAMES
    style_df = data_df.loc[[idx for idx in data_df.index if idx in label_set],: ].copy()
    data_df  = data_df.loc[[idx for idx in data_df.index if not idx in label_set],: ].copy()
    #
    names = sorted ( list( set( style_df.loc[ sample_name_label,:].values )) )
    #
    # CREATE THE FACTORS FOR THE X AXIS
    factors = np.array([ [ (ef,time) for time in pairing_order] for ef in effect_order ]).reshape(-1,2)
    factors = [ tuple(f) for f in factors ]
    #
    # CREATE A LOOKUP FOR THE SAMPLES
    se_lookup = { s:e for (s,e) in zip(*(style_df.loc[[sample_name_label,effect_label],:].values)) }
    from bokeh.io import show, output_file
    from bokeh.models import FactorRange
    from bokeh.plotting import figure
    #
    ymax = np.max( data_df.loc[feature_name].values )
    ymin = np.min( data_df.loc[feature_name].values )
    #
    ttips = [   ("index "  , "$index"   ) ,
                ("(x,y) "  , "(@x, @y)" ) ,
                ("name "   , "@name"    ) ]
    #
    hover = HoverTool ( tooltips = ttips )
    #
    p = figure( x_range = FactorRange(*factors) ,
                #y_range = [ymin,ymax],#Range1d( *[ymin-np.sign(ymin)*0.1*ymin , ymax+np.sign(ymax)*0.1*ymax] ),
                plot_height = plot_height , toolbar_location = 'right' ,
                tools = [ hover,'box_zoom','wheel_zoom','pan','reset','save' ],
                title = title_prefix + str(feature_name) )

    if hue_colors is None :
        hue_colors = { h:c for h,c in zip(hue_order,colors_[:len(hue_order)]) }

    if not case_label is None :
        # AVAILABLE BOKEH PLOTTING TOOLS
        mappings_ = [  p.circle , p.circle_cross , p.circle_x ,
              p.diamond , p.diamond_cross , p.square , p.square_cross , p.square_x ,
              p.triangle , p.inverted_triangle , p.asterisk , p.cross , p.x , p.dash  ]
        case_mapping = { h:c for h,c in zip(case_order,mappings_[:len(case_order)]) }
        print( [ co +' <=> '+ str(cm).split('method ')[1].split(' of')[0] for cm,co in zip( case_mapping.values() , case_order ) ] )

    if True :
        for name in names :
            X = [ (se_lookup[idx[0]],idx[1]) for idx in data_df.loc[ feature_name,[name] ].index ]
            Y = data_df.loc[ feature_name,[name] ].values
            if bVerbose :
                print(name,X,2**Y-1)
            if not hue_label is None :
                hue_val = list(set(style_df.loc[hue_label,name].values))[0]
                color,clabel = hue_colors[hue_val],hue_val
            else :
                color,clabel="red" , None
            dname = [ name for i in range(len(X)) ] # clabel
            p .line( x=X , y=Y , color=color , name=name , line_width=2 )
            if not case_label is None :
                case_val = list(set(style_df.loc[case_label,name].values))[0]
                # CALL THE PLOTTING TOOL # case_val
                case_mapping[case_val]( x=X , y=Y , line_color=color , name=name, fill_color="white", size=icon_size )
            else :
                p.circle( x=X , y=Y , line_color=color , fill_color="white", size=icon_size , name=name )

    p .y_range.start = 0
    p .x_range.range_padding = 0.1
    p .xaxis.major_label_orientation = 1
    p .xgrid.grid_line_color = None

    p .grid.grid_line_width = 0
    p.title.text_font_size  = major_label_text_font_size[0]

    p.xaxis.group_text_font_size       = minor_label_text_font_size[0]
    p.xaxis.axis_label_text_font_size  = minor_label_text_font_size[0]
    p.xaxis.major_label_text_font_size = minor_label_text_font_size[1]

    p.yaxis.axis_label  = yaxis_label
    p.yaxis.axis_label_text_font_size  = minor_label_text_font_size[0]
    p.yaxis.major_label_text_font_size = minor_label_text_font_size[1]

    p.output_backend = 'webgl'

    p.y_range = Range1d( ymin*0.95, ymax*1.05 )

    if pReturn :
        return ( p )
    show( p )


def example_dynamic_linking( bCLI=True ):
    
    desc__="""        scatter2boxplot( ["P1","P2","P3"], [10.0,7.0,12.0], [2.0,4.0,16.5], {'funny':["yes","no","maybe"]},
            [('','Kalle','0','0'),('','Kalle','1','1'),('','Stina','0','2'),('','Stina','1','3'),('','Jens','0','2'),('','Jens','1','3'),
             ('','Anna','0','0'),('','Anna','1','1'),('','Adam','0','0'),('','Adam','1','1'),('','Niklas','0','2'),('','Niklas','1','3'),
             ('','Lina','0','2'),('','Lina','1','3'),('','Stina','0','0'),('','Stina','1','1')],
            [[ 3.0,3.1,5.2,10.2,7.0,17.1,2.0,2.5,1.5,1.8,6.1,16.2,7.4,16.3,3.2,2.4 ],
             [ 4.1,14.0,2.3,2.3,2.0,3.0,2.0,13.1,4.0,12.2,4.2,14.0,2.1,2.2,4.5,15.5],
             [ 14.1,2.1,2.2,2.1,3.2,3.4,12.5,2.5,15.6,5.6,1.2,2.0,2.2,2.2,11.1,2.1 ] ],
            [125.2,125.2,75.0,75.0,75.0,75.0,175.0,175.0,135.2,135.2,65.0,65.0,55.0,55.0,195.0,195.0],
            [0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0] ,
            {'x1':'x axis','y1':'y axis','x2':'x axis','y2':'y axis', 0: ('healthy',"blue"), 1: ('sick','red')} ,
            categories={ '0':'Basal-H', '1':'30min-H','2':'Basal-S','3':'30min-S'},
            patient_pos=1, category_pos=3,
            bShow=False )"""

    if not bCLI :
        scatter2boxplot( ["P1","P2","P3"], [10.0,7.0,12.0], [2.0,4.0,16.5], {'funny':["yes","no","maybe"]},
            [('','Kalle','0','0'),('','Kalle','1','1'),('','Stina','0','2'),('','Stina','1','3'),('','Jens','0','2'),('','Jens','1','3'),
             ('','Anna','0','0'),('','Anna','1','1'),('','Adam','0','0'),('','Adam','1','1'),('','Niklas','0','2'),('','Niklas','1','3'),
             ('','Lina','0','2'),('','Lina','1','3'),('','Stina','0','0'),('','Stina','1','1')],
            [[ 3.0,3.1,5.2,10.2,7.0,17.1,2.0,2.5,1.5,1.8,6.1,16.2,7.4,16.3,3.2,2.4 ],
             [ 4.1,14.0,2.3,2.3,2.0,3.0,2.0,13.1,4.0,12.2,4.2,14.0,2.1,2.2,4.5,15.5],
             [ 14.1,2.1,2.2,2.1,3.2,3.4,12.5,2.5,15.6,5.6,1.2,2.0,2.2,2.2,11.1,2.1 ] ],
            [125.2,125.2,75.0,75.0,75.0,75.0,175.0,175.0,135.2,135.2,65.0,65.0,55.0,55.0,195.0,195.0],
            [0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0] ,
            {'x1':'x axis','y1':'y axis','x2':'x axis','y2':'y axis', 0: ('healthy',"blue"), 1: ('sick','red')} ,
            categories={ '0':'Basal-H', '1':'30min-H','2':'Basal-S','3':'30min-S'},
            patient_pos=1, category_pos=3,
            bShow=False )
    else :
        print ( desc__ )


def example_lineplot( bCLI=True ):
    desc__ = """
    #
    # WE DEFINE SOME SAMPLES , AN EXPERIMENT AND AN EFFECT
    #
    identities = { 10 : ['a','b','c','d','e','f','g','h','i','j'] }
    timepoints = {  3 : ['Basal','Intervention','Rest'] }
    #
    effects    = {  2 : {'Sick':{'c','d','i','j','f'},'Healthy':{'a','b','e','g','h'} } }
    s_effects  = invert_dictlist( effects )
    #
    cases      = {  2 : {'Treadmill':{'c','d','e','g','h'},'Cycle':{'a','b','i','j','f'} } }
    s_cases    = invert_dictlist( cases )
    #
    genders      = {  2 : {'Male':{'c','d','i','j','h'},'Female':{'a','b','e','g','f'} } }
    s_genders    = invert_dictlist( genders )
    #
    n_samples  = list(identities.keys())[0] * list(timepoints.keys())[0]
    n_features = 10
    #
    print ( n_samples , n_features )
    print ( list(identities.values()) , list(timepoints.values()) )
    #
    # CREATE A NAMED DATAFRAME
    column_names = [ [ it + '_' + t for it in list(identities.values())[0] ] \
                       for t in list(timepoints.values())[0] ]
    column_names = list(np.array(column_names).reshape(1,-1))[0]
    column_names = [ s_effects[ c.split('_')[0] ] + '_' + c + '_' + s_cases[ c.split('_')[0] ] + '_' + s_genders[ c.split('_')[0] ] for c in column_names ]
    #
    rand_df = pd.DataFrame( np.random.rand( n_samples * n_features ) .reshape( n_features , n_samples ) )
    rand_df .columns = column_names
    rand_df .loc[ 'Effect_[str]' ] = [c.split('_')[0] for c in rand_df.columns]
    rand_df .loc[ 'Times_[str]'  ] = [c.split('_')[2] for c in rand_df.columns]
    rand_df .loc[ 'Sample_[str]' ] = [c.split('_')[1] for c in rand_df.columns]
    rand_df .loc[ 'Case_[str]'   ] = [c.split('_')[3] for c in rand_df.columns]
    rand_df .loc[ 'Gender_[str]'   ] = [c.split('_')[4] for c in rand_df.columns]
    pairing_order = ['Rest','Basal','Intervention']

    print ( rand_df )

    create_paired_figure_p ( rand_df ,
            sample_name_label = 'Sample_[str]' ,
            pairing_label     = 'Times_[str]'  ,
            hue_label         = 'Case_[str]'   ,
            case_label        = 'Gender_[str]' ,
            effect_label      = 'Effect_[str]' ,
            pairing_order     = pairing_order  )

    """
    print ( desc__ )
    if not bCLI:
        return
    #
    # WE DEFINE SOME SAMPLES , AN EXPERIMENT AND AN EFFECT
    #
    identities = { 10 : ['a','b','c','d','e','f','g','h','i','j'] }
    timepoints = {  3 : ['Basal','Intervention','Rest'] }
    #
    effects    = {  2 : {'Sick':{'c','d','i','j','f'},'Healthy':{'a','b','e','g','h'} } }
    s_effects  = invert_dictlist( effects )
    #
    cases      = {  2 : {'Treadmill':{'c','d','e','g','h'},'Cycle':{'a','b','i','j','f'} } }
    s_cases    = invert_dictlist( cases )
    #
    genders      = {  2 : {'Male':{'c','d','i','j','h'},'Female':{'a','b','e','g','f'} } }
    s_genders    = invert_dictlist( genders )
    #
    n_samples  = list(identities.keys())[0] * list(timepoints.keys())[0]
    n_features = 10
    #
    print ( n_samples , n_features )
    print ( list(identities.values()) , list(timepoints.values()) )
    #
    # CREATE A NAMED DATAFRAME
    column_names = [ [ it + '_' + t for it in list(identities.values())[0] ] \
                       for t in list(timepoints.values())[0] ]
    column_names = list(np.array(column_names).reshape(1,-1))[0]
    column_names = [ s_effects[ c.split('_')[0] ] + '_' + c + '_' + s_cases[ c.split('_')[0] ] + '_' + s_genders[ c.split('_')[0] ] for c in column_names ]
    #
    rand_df = pd.DataFrame( np.random.rand( n_samples * n_features ) .reshape( n_features , n_samples ) )
    rand_df .columns = column_names
    rand_df .loc[ 'Effect_[str]' ] = [c.split('_')[0] for c in rand_df.columns]
    rand_df .loc[ 'Times_[str]'  ] = [c.split('_')[2] for c in rand_df.columns]
    rand_df .loc[ 'Sample_[str]' ] = [c.split('_')[1] for c in rand_df.columns]
    rand_df .loc[ 'Case_[str]'   ] = [c.split('_')[3] for c in rand_df.columns]
    rand_df .loc[ 'Gender_[str]'   ] = [c.split('_')[4] for c in rand_df.columns]
    pairing_order = ['Rest','Basal','Intervention']

    print ( rand_df )

    create_paired_figure_p ( rand_df ,
            sample_name_label = 'Sample_[str]' ,
            pairing_label     = 'Times_[str]'  ,
            hue_label         = 'Case_[str]'   ,
            case_label        = 'Gender_[str]' ,
            effect_label      = 'Effect_[str]' ,
            pairing_order     = pairing_order  )

    
if __name__=='__main__':

    example_dynamic_linking ( False )
    example_lineplot ( False )
    #TODO : ADD MORE CUSTOMIZABILITY FOR USERS. MAKE A DYNAMICALLY LINKED LINEPLOT



