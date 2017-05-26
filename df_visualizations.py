import pandas  as pd
import numpy   as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.cm     as cm

# Do cool correlation plot, with scatter and histograms
def corr_plot( inp_df, exclude = None, focus = None, ordered=True ):

    
    col_list = inp_df.columns.values
    
    # Remove elements we are excluding
    if ( exclude != None ):
        if type( exclude ) is list:
            for element in exclude:
                index    = np.argwhere( col_list==element )
                col_list = np.delete(   col_list, index   )
        else:
            index    = np.argwhere( col_list==element )
            col_list = np.delete(   col_list, index   )


            
    # Put at the bottom, for easy comparison
    if ( focus != None ):

        if (ordered):
            col_list = inp_df.corrwith( inp_df[focus]).sort_values( ascending=False ).index
        else:
            index    = np.argwhere( col_list==focus )
            col_list = np.delete(   col_list, index )
            col_list = np.insert( col_list, 0, focus )

    df     = inp_df[col_list].copy()    
    
    corr   = df.corr( method='spearman' )
    corr_v = corr.as_matrix()
    
    # Mask the upper right so it can't be seen
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    mask = np.transpose( mask )
    

    # Plot the correlation with color background in upper right
    cmap = cm.get_cmap('coolwarm')
    axes = pd.tools.plotting.scatter_matrix( df )
    for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
        axes[i, j].cla()
        axes[i, j].set_axis_bgcolor( cmap( 0.5 * corr_v[i,j] + 0.5) ) 
        axes[i, j].annotate("%0.3f" %corr_v[i,j], (0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
    plt.show()
    
# Can do histogram with strings
def hist_plot( inp_df, col ):
    df = inp_df.groupby(col).size()
    ax = df.plot(kind='bar', title=col )
    ax.set_xlabel( ' ' )
    ax.set_ylabel( 'Count' )
    plt.show()
    
# Bar plot for column taking average for y col
def plot_avg( inp_df, x_col, y_col ):
    
    df = inp_df[[x_col,y_col]].copy()
    
    means = df.groupby( x_col ).mean()
    std   = df.groupby( x_col ).std().fillna(0)

    ax = means.plot(kind='bar' , title=x_col, yerr=std, legend=False )
    ax.set_xlabel( ' ' )
    ax.set_ylabel( y_col )
    plt.show()