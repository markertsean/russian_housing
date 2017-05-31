import numpy  as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone



# Fill nans with most common values
def fill_nans_mode( inp_df, column ):
    modeVal =  inp_df[ column ].mode()
    return     inp_df[ column ].fillna( modeVal )

# Normalize index
def normalize_column( inp_df, column, maxVal=None, minVal=None ):
    
    new_column = inp_df.copy()
    
    max_value = maxVal
    min_value = minVal
    
    if( max_value == None ):
        max_value = inp_df[column].max()
    if( min_value == None ):
        min_value = inp_df[column].min()
        
    new_column[column] = ( inp_df[ column ] - float(min_value) ) / ( max_value - min_value )
    new_column.ix[ new_column[column]<0, column ] = 0.0
    new_column.ix[ new_column[column]>1, column ] = 1.0

    return 2*new_column[column]-1.

# Find normalization values within input sigma, returns normalization parameters that ignores outliers
def normalize_column_sigma( inp_df, column, lower_bound=True, upper_bound=True, n_sigma=3.0 ):
    
    new_column = inp_df[column].copy()
    
    myMean = new_column.mean()
    myStd  = new_column.std()
    
    if ( lower_bound ):
        new_column =  new_column[ new_column > ( myMean - n_sigma * myStd ) ]
        
    if ( upper_bound ):
        new_column =  new_column[ new_column < ( myMean + n_sigma * myStd ) ]
        
    return normalize_column( inp_df, column, minVal=new_column.min(), maxVal=new_column.max())

# Normalize index
def scale_column( inp_df, column, maxVal=None, minVal=None ):
    
    new_column = inp_df.copy()
    
    max_value = maxVal
    min_value = minVal
    
    if( max_value == None ):
        max_value = inp_df[column].max()
    if( min_value == None ):
        min_value = inp_df[column].min()
        
    new_column[column] = ( inp_df[ column ] - float(min_value) ) / ( max_value - min_value )
    new_column.ix[ new_column[column]<0, column ] = 0.0
    new_column.ix[ new_column[column]>1, column ] = 1.0

    new_column[column] = 2*new_column[column]-1.
    
    
    return ( new_column[column] - new_column[column].mean() ) / new_column[column].std()

# Find z scale values within input sigma, returns normalization parameters that ignores outliers
def scale_column_sigma( inp_df, column, lower_bound=True, upper_bound=True, n_sigma=3.0 ):
    
    new_column = inp_df[column].copy()
    
    myMean = new_column.mean()
    myStd  = new_column.std()
    
    if ( lower_bound ):
        new_column =  new_column[ new_column > ( myMean - n_sigma * myStd ) ]
        
    if ( upper_bound ):
        new_column =  new_column[ new_column < ( myMean + n_sigma * myStd ) ]
        
    return scale_column( inp_df, column, minVal=new_column.min(), maxVal=new_column.max())

# Generate PCA, and return relevant columns
def generate_reduced_PCA( inp_df             , # Dataframe to play with
                          N_c                , # Number of columns to accept from PCA
                          cols=[]            , # List of column names
                          contain_str = None , # Common string in names, to find in function
                          col_names   = None , # String to append to columns, if no contain_str
                          corr_df     = None , # If doing verbose, the data series to run correlation against
                          verbose     = False, # Whether to print stuff as go along 
                          normalize       = False ,  # Perform linear normalization
                          maxVal          = None  ,  # Arguments  for normalization
                          minVal          = None  ,  # Arguments  for normalization
                          sigma_normalize = False ,  # Perform normalization, ommiting outliers
                          lower_bound     = True  ,  # Arguments  for sigma normalization 
                          upper_bound     = True  ,  # Arguments  for sigma normalization 
                          n_sigma         = 3.0      # Arguments  for sigma normalization
                        ):
    # Must have some indication of columns to use
    if ( (contain_str == None) and 
         (len(cols)   == 0   ) ):
        print 'Error in generate_reduced_PCA:'
        print '  Must provide cols (list of columns), or contain_str (common component of column names)'
        return
        
    # Will use input columns, unless contain str provided (for reducing columns with redundant names)
    if (  contain_str != None   ):
        cols = inp_df.ix[:, inp_df.columns.str.contains( contain_str ) ].columns.values

    if ( N_c > len(cols) ):
        print 'Error in generate_reduced_PCA:'
        print '  number of components exceeds number of valid columns'
        return        

        
    foo = inp_df[cols].copy()
    
    
    # Normalize if not already normalized
    if ( normalize or sigma_normalize ):
        for col in foo.columns.values:
            if ( normalize ):
                foo[col] = scale_column      ( foo, col, maxVal, minVal )
            else:
                foo[col] = scale_column_sigma( foo, col, lower_bound, upper_bound, n_sigma )
        
    
    # Generate column names for returning series
    new_cols = []
    for i in range( 0, N_c ):
        front = ''
        if ( contain_str != None ):
            front = contain_str+'_'
        if (  col_names  != None ):
            front = col_names+'_'
        new_cols.append( front+'pca_'+str(i) )

    # Actually do the PCA
    my_pca = PCA( n_components = N_c )
    my_pca.fit( foo )

    bar = pd.DataFrame( my_pca.transform( foo ), columns=new_cols )
    
    if ( verbose ):
        print 'Using columns:'
        print cols
        print ''
        
        print 'Using ',N_c,' components, variance ratio:'
        print my_pca.explained_variance_ratio_
        print 'Total explained variance: ', my_pca.explained_variance_ratio_.sum()
        print ''
        
        print 'New columns:'
        print new_cols
        
        # Correlation with another series against each component
        if isinstance( corr_df, pd.Series ):
            print ''
            print 'Correlation with input series:'
            print bar.corrwith( corr_df )
            print ' '
        
    return bar


# Returns column with values replaced by integers
def numerize_col( inp_df, col ):

    new_dict = {}
    counter  = 0

    for item in inp_df[col].unique():
        new_dict[item] = counter
        counter        = counter + 1

    return inp_df[col].replace( new_dict )


# Change the classifications to multiple binary columns
def binary_classification( inp_df, ignore=None, lowerLim=2, upperLim=20 ):

    global bin_class_array
    
    bar=inp_df.copy()
    col_list = bar.columns.values
    
    if ( ignore != None ):
        if type( ignore ) is list:
            for element in ignore:
                index    = np.argwhere( col_list==element )
                col_list = np.delete(   col_list, index )
        else:
            index    = np.argwhere( col_list==ignore )
            col_list = np.delete( col_list, index )
            
    # Consider each column
    for col in col_list:
        options = sorted(bar[col].unique()) # Possible options

        # Only reformat the classification columns
        if ( len(options) > lowerLim and 
             len(options) < upperLim ):
            iterator = 0
            # Create new binary clssifier for each class column
            for item in bar[col].unique():
                
                new_col = col+'_'+str( item )
                bar[new_col] = 0
                bar.ix[ bar[col]==item, [new_col] ] = 1

            # Remove previous classification
            bar.drop( col, axis=1, inplace=True )
    return bar

def run_kfold( clf, train_x_df, train_y_df, nf=10 ):
    
    kf = KFold( n_splits=nf )
    kf.get_n_splits( train_x_df )
    
    outcomes = []
    fold = 0
    
    # Generate indexes for training and testing data
    for train_index, test_index in kf.split( train_x_df ):
        
        fold += 1
        
        # Generate training and testing data from input sets
        x_train = train_x_df[train_index]
        y_train = train_y_df[train_index]
        x_test  = train_x_df[test_index]
        y_test  = train_y_df[test_index]
        
        
        # Foooooooo
        clf.fit( x_train, y_train )
        predictions = clf.predict( x_test )
        accuracy = r2_score( y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))
        
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0} +/- {1}".format(mean_outcome,np.std(outcomes)))
    
def optimize_fit( clf, train_x, train_y, grid_params, nf=10, verbose=True ):
    
    kf = KFold( n_splits=nf )
    kf.get_n_splits( train_x )
    
    outcomes = []
    clf_list = []
    fold = 0
    
    # Generate indexes for training and testing data
    for train_index, test_index in kf.split( train_x ):
        
        fold += 1
        
        # Generate training and testing data from input sets
        x_train = train_x[train_index]
        y_train = train_y[train_index]
        x_test  = train_x[test_index]
        y_test  = train_y[test_index]
        
        
        # 
        
        new_clf = GridSearchCV( clf, grid_params ) 
        
        new_clf.fit( x_train, y_train )
        predictions = new_clf.predict( x_test )
        accuracy = r2_score( y_test, predictions )
        outcomes.append(accuracy)
        
        clf_list.append( clone( new_clf.best_estimator_ ) )
        if ( verbose ):
#            print("Fold {0} accuracy: {1}".format(fold, accuracy)), ', ', new_clf.best_score_, new_clf.best_params_
            print "Fold %2i accuracy: %6.4f " % (fold, accuracy), ', ', '%6.4f '%new_clf.best_score_, new_clf.best_params_
        
    best_clf_score = 0
    best_clf_index = 0
    best_clf_acc   = 0
    
    clf_num = 0

    print ' '

    # Check each winning CLF against the group
    # and pick the best of the best
    for test_clf in clf_list:
        
        accuracies = []
        fold       = 0
        
        for train_index, test_index in kf.split( train_x ):
        
            fold += 1

            # Generate training and testing data from input sets
            x_train = train_x[train_index]
            y_train = train_y[train_index]
            x_test  = train_x[test_index]
            y_test  = train_y[test_index]
        
            test_clf.fit( x_train, y_train )

            predictions = test_clf.predict( x_test )
            accuracy = r2_score( y_test, predictions )
            accuracies.append(accuracy)
        
        mean_outcome = np.mean( accuracies )
        print "Clf %2i Mean Accuracy: %6.4f +/- %6.4f" % (clf_num,mean_outcome,np.std(accuracies))

        # Figure out which clf is the best
        if ( mean_outcome > best_clf_score ):
            best_clf_index = clf_num
            best_clf_score = mean_outcome
            
        clf_num = clf_num + 1
    
    # Fit and return best fit
    ret_clf = clf_list[ best_clf_index ]
    ret_clf.fit( train_x, train_y )
    
    if ( verbose ):
        print 'Using CLF with accuracy: %10.6f' % best_clf_score
        print 'CLF params: ', ret_clf.get_params( deep=False )
    
    
    return ret_clf