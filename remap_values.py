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