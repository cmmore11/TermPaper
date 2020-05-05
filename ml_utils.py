"""

- machine learning functions:
   classification algorithms
   evaluate classification performance
   regression algorithms

    @author thgoebel, University of Memphis
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import struct, os
#=========================================================================
#                   data preprocessing, splitting load data
#=========================================================================
from numpy.core.multiarray import ndarray


def splitData( mData, splitRatio, **kwargs):
    """
    split into training and test data according to split ratio
    :param mData:
    :param splitRatio:
           kwargs['random'] = True, random split; detaulf=False
                                    split into first and second part by default
    :return: mTrainData, mTestData
    """
    # total instances
    Ntot   = mData[0].shape[0]
    Ntrain = int( round(Ntot*splitRatio))
    Ntest  = Ntot - Ntrain
    if 'random' in kwargs.keys() and kwargs['random'] == True:
        # create random boolean vector for training data
        aID = np.arange( Ntot, dtype = int)
        np.random.shuffle( aID)
    else:
        aID =  np.arange( Ntot, dtype = int)
    aIDtrain = np.array( sorted( aID[0:Ntrain]))
    aIDtest  = np.array( sorted( aID[Ntrain::]))
    return mData[:,aIDtrain],mData[:,aIDtest]

def str2int( a_y):
    """
    -change elements in a_y from strings to integer
    :param a_y:  - e.g. class labels or attribute vector
    :return:
           a_y_int - shape(a_y)
    """
    a_y_int = np.zeros( len(a_y), dtype = int)
    a_lab = np.unique( a_y)
    n_lab = len(a_lab)
    for i in range( n_lab):
        a_y_int[a_lab[i]==a_y] = i
    return a_y_int


def onehot( a_y):
    """ Encode class labels in y into one-hot representation
    Parameters
    -----------
    y : array, shape = [n_examples]
        target classes (or attribute vector)

    :return
    ----------
    onehot : array, shape = (n_examples, n_labels)
    """
    a_lab = np.unique( a_y)
    n_lab = len(a_lab)
    m_y_int = np.zeros((n_lab, a_y.shape[0]))
    for i in range( n_lab):
        m_y_int[i,a_lab[i]==a_y] = 1
    return m_y_int

    # n_classes = len( np.unique( y))
    # m_OH = np.zeros(( n_classes, y.shape[0]))
    # for idx, val in enumerate( y.astype( int)):
    #     m_OH[val, idx] = 1.
    # return m_OH.T
#-------------------------------1----------------------------------------
#                           data I/O
#------------------------------------------------------------------------

def number2matrix( **kwargs):
    """
    --> create binary matrix that represents, noisy handwritten number
    --> this function first creates a random number between 0 to 9, saves it a low res. .png
        the image is loaded, converted to binary matrix + noise
    optional arguments:
    :fracNoise  - how much noise should be addedd (fow low res ~ 5%)
    :dpi        - resolution of image - default 10
    :binThres   - 10 - used to convert gray scale (0 to 255) to binary 0 or 1  
    return: mNo, numClass - binary matrix representing number, number label / class
    """
    fracNoise = 0.08
    dpi       = 10
    # everything above is set to 0, below to 1
    binThres  = 10 # set threshold to convert gray to binary
    if 'binThres' in kwargs.keys() and kwargs['binThres'] is not None:
        binThres = kwargs['binThres']
    if 'dpi' in kwargs.keys() and kwargs['dpi'] is not None:
        dpi = kwargs['dpi']
    if 'fracNoise' in kwargs.keys() and kwargs['fracNoise'] is not None:
        fracNoise = kwargs['fracNoise']
    #---------------------------------1---------------------------------------
    #                     create number write to file
    #-------------------------------------------------------------------------
    # creates an array, use index to change to scalar
    ranNo = np.random.random_integers( 0, 9, 1)[0]
    file_name = "dummy_%.2f.png"%( fracNoise)
    plt.figure(1, figsize=(1,1))
    plt.text( .01, .01, str( ranNo), fontsize = 100)
    plt.axis('off')
    plt.savefig( file_name, bbox_inches='tight', transparent=True, dpi = dpi)
    #plt.show()
    plt.clf()

    #---------------------------------2---------------------------------------
    #                       load figure, add noise
    #-------------------------------------------------------------------------
    # load image as gray scale, default is 4 coulmn rgb with last column = alpha
    mNo = scipy.ndimage.imread( file_name, mode='L')
    # convert to binary
    sel = mNo > binThres
    mNo[sel] = 0
    mNo[~sel]= 1
    print( 'curr. ran number', ranNo, 'image size: ', mNo.shape[0]*mNo.shape[1], mNo.shape[0], mNo.shape[1])
    # add random noise
    Nran = int( mNo.shape[0]*mNo.shape[1]*fracNoise)
    ix   = np.random.random_integers( 0, mNo.shape[0]-1, Nran)
    iy   = np.random.random_integers( 0, mNo.shape[1]-1, Nran)
    # flip 0 and 1 at random locations
    mNo[ix,iy] = 1 - mNo[ix,iy]
    return mNo,ranNo

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`
    Parameters
    ------------
    path: str
            specify data directory if different for cwd
    kind: train (train or t10k)
          switch between training or testing data (10k numbers)

    """
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape( len(labels), 784)
        images = ((images / 255.) - .5) * 2
    return images, labels


def loadTab2dic( file_in, **kwargs):
    """
    :param file_in: - read attributes from file, assuming all attributes have nominal values
    :param kwargs:

    :return: mostFrequentAttribute, lRule ( = [attrValue, class, nCountErr, nTotInst]
    """
    comment   = '#'
    delimiter = None
    if 'comment' in kwargs.keys() and kwargs['comment'] is not None:
        comment = kwargs['comment']
    if 'delimiter' in kwargs.keys() and kwargs['delimiter'] is not None:
        delimiter = kwargs['delimiter']
    import os
    # read header line which contains the attibute names
    with open( file_in, 'r') as file_obj:
        lAtt = file_obj.readline().strip().split()

    # remove comment str from attribute names
    lAtt = [i.strip( comment) for i in lAtt]

    mData = np.genfromtxt( file_in, dtype=str , comments = comment, delimiter = delimiter).T
    dData = {}
    i = 0
    for tag in lAtt:
        dData[tag] = np.array( mData[i])
        i += 1
    return dData


#------------------------------2------------------------------------------
#                     Baseline Algorithm
#-------------------------------------------------------------------------
def listValueClass( dData, classStr):
    """
    - for each unique attribute value list class and count

    :param dData:
    :param classStr:
    :return:
    """
    # list of all attributes
    lAttribute = dData.keys()
    lAttribute.remove( classStr)#exclude last column with  classes
    lAttAll, lValAll, lClassAll, lN_all = [],[],[],[]
    # each attribute
    for sAtt in lAttribute:
        aInstance = np.unique( dData[sAtt])
        # all instances with current attribute value
        i_inst = 0
        for sValue in aInstance:
            sel = sValue == dData[sAtt]
            # count instances for each attribute and class
            aUniClass, aN = np.unique( dData[classStr][sel], return_counts = True)
            # count number of occurrence of each attribute value and class
            for iN in range( len( aN)):
                #print sAtt, sValue, aUniClass[iN], aN[iN]
                lAttAll.append( sAtt)
                lValAll.append( sValue)
                lClassAll.append( aUniClass[iN])
                lN_all.append( aN[iN])
    # select most frequent attribute element and class
    iMax = lN_all.index( max( lN_all))
    return lAttAll[iMax], [ lValAll[iMax],lClassAll[iMax], lN_all[iMax]]

def random_label( train, n_test):
    """
    :param train: class labels in training data set
    :param n_test: size of test data
    :return: randomized training class labels
    """
    a_ID = np.random.randint( 0, n_test-1, n_test)
    return train[a_ID]

def zeroR( dData, classStr):
    """
    simply predict majority class every time

    :param dData:    - dictionary with nominal data entries
    :param classStr: - class identifier
    :return: lRule (lRule = [class, nCount])
    """
    # list of all classes
    lClassUni, aN = np.unique( dData[classStr], return_counts=True)
    #print lClassUni, aN
    # select most frequent attribute element and class
    iMax =  aN.argmax()
    return [lClassUni[iMax], aN[iMax]]

def zeroR_num( mData, n_test):
    """zero r for numeric data"""
    aUni, aN =  np.unique( mData[-1], return_counts=True)
    mostFre = aUni[aN == aN.max()][0]
    return np.ones( n_test)*mostFre

def oneR_num( m_data, a_y, m_test):
    """
    - one rule algorithm for numerical data
        --> training and prediction step are done simultaneously
    For each predictor,
         For each value of that predictor, make a rule as follows;
               Count how often each value of target (class) appears
               Find the most frequent class
               Make the rule assign that class to this value of the predictor
         Calculate the total error of the rules of each predictor
    Choose the predictor with the smallest total error.
    :param mData:   - training data
    :param a_y:      - training class labels
    :param m_test:   - features in test data set
    :return: a_y_hat - predicted class labels for m_test
    """
    a_tot_err  = np.zeros( m_data.shape[0])
    d_pred     = {} #keep track of prediction for each attr.
                    # numerical from i=1 to n
    for i in range( m_data.shape[0]):
        ##A## count how often each instance appears - DENOMINATOR
        aInsUni, aIdIns, aN_ins = np.unique( m_data[i],return_index=True, return_counts=True)
        ##B## Count how often each value of target (class) is predicted
        a_n_err_attr = np.zeros( len( aInsUni))
        a_cl_pred  = np.zeros( len( aInsUni))
        for j in range( len( aInsUni)):
            sel_ins = aInsUni[j] == m_data[i]
            # count no. of unique class attribute pairs -  NUMERATOR
            aClUni, aIdCl, aN_cl = np.unique(a_y[sel_ins], return_index=True, return_counts=True)
            i_max = np.argmax( aN_cl)
            a_n_err_attr[j] = aN_ins[j] - aN_cl[i_max]
            a_cl_pred[j]  = aClUni[i_max]
        ##C## compute total error as sum of incorrect classifications over no. of class elements
        a_tot_err[i] = a_n_err_attr.sum()/aN_ins.sum()
        d_pred[str(i)] = np.array([  aInsUni, a_cl_pred])
    #----------------------prediction step------------------------------------
    # find matching attribute instance and corresponding prediction
    sel = a_tot_err == a_tot_err.min()
    s_BestAttr = np.array( list( d_pred.keys()))[sel][0]
    ## attribute instances in test data
    a_attr_test = m_test[int(s_BestAttr)]

    a_y_hat = np.zeros( len( a_attr_test))
    for i_y in range( len( a_y_hat)):
        # find closest sample match in case the same sample is not present in feature
        sel = abs( a_attr_test[i_y] - d_pred[s_BestAttr][0]) == abs( a_attr_test[i_y] - d_pred[s_BestAttr][0]).min()
        if sel.sum() > 1:
            a_y_hat[i_y] = d_pred[s_BestAttr][1][sel][0]
        else:
            a_y_hat[i_y] = d_pred[s_BestAttr][1][sel]
    # print( 'best attribute', s_BestAttr)
    # print( 'rule', d_pred[s_BestAttr])
    # print( 'test data', a_attr_test)
    # print( 'prediction', a_y_hat)
    return a_y_hat, d_pred[s_BestAttr]

def oneR( dData, classStr, **kwargs):
    """
    OneR Algorithm:
    For each predictor,
         For each value of that predictor, make a rule as follows;
               Count how often each value of target (class) appears
               Find the most frequent class
               Make the rule assign that class to this value of the predictor
         Calculate the total error of the rules of each predictor
    Choose the predictor with the smallest total error.
    :param dData:    - dictionary with nominal data entries
    :param classStr: - class identifier

    :param kwargs:

    :return: [attributeStr], oneRule=[ [attValue, class, errCount, nTotInstancesWithValue]]
    """
    dRule = {}
    # list of all attributes
    lAttribute = list( dData.keys())
    lAttribute.remove( classStr)#exclude last column with  classes
    # keep track of total error rate for each attribute
    lTotErr = []
    # each attribute
    for sAtt in lAttribute:
        aInstance = np.unique( dData[sAtt])
        # create list y inside of dRule dic with nRows = len(aInstance)
        dRule[sAtt] = []
        nTotErr     = 0
        # all instances with current attribute value
        i_inst = 0
        for curr_inst in aInstance:
            sel = curr_inst == dData[sAtt]
            # count instances for each attribute and class
            aUniClass, aN = np.unique( dData[classStr][sel], return_counts = True)
            # error rate for  attribute value associated with most frequent class
            selMax = aN == aN.max()
            errCount = sel.sum() - aN.max()
            # create rule for most frequent class
            dRule[sAtt].append( [curr_inst,  aUniClass[selMax][0], errCount, sel.sum()])
            nTotErr += errCount
            i_inst  += 1
        # total error rate for each attribute
        fErrTot = nTotErr/len( dData[sAtt])
        lTotErr.append(  fErrTot)
    #
    #return best rules (lowest err rate attribute)
    #
    minTotErr = min( lTotErr)
    # get indices for all list entries with min value
    iMinErr = [i for i, j in enumerate( lTotErr) if j == minTotErr]
    # index of first value == min. value
    #iMin = lTotErr.index( min( lTotErr))
    if len( iMinErr) > 1:
        print('several attributes have the same min err rate: ', np.array(lAttribute)[iMinErr], np.array(lTotErr)[iMinErr])
        print('use first attribute')
    print('------------------ZERO-R - classification rule---------------------------')
    print('Feature:', lAttribute[iMinErr[0]], dRule[lAttribute[iMinErr[0]]], 'err. rate: ',np.array(lTotErr)[iMinErr[0]])
    print()
    return lAttribute[iMinErr[0]], dRule[lAttribute[iMinErr[0]]]

#---------------------------------3---------------------------------------
#                       algorithm performance using test data
#-------------------------------------------------------------------------

def bin_confMatrix( lPred, lObs, posStr, negStr):
    """
    compute confusion matrix based on predicted and observed class elements
    binary confusion matrix contains: TP    FN
                                      FP    TN
                                      -accuracy
                                      -sensitivity (recall) -the proportion of actual positive cases which are correctly identified.
                                      -specificity - the proportion of actual negative cases which are correctly identified
    :param lPred:  - prediction class values
    :param lObs:   - observed class value
           posStr  - 'Yes', 'True',  '1' etc  string that denotes positive outcome
           negStr  - 'No',  'False', '0' etc  string that denotes positive outcome
    :return:
    """
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range( len(lPred)):
        if lPred[i] == lObs[i]:
            if lPred[i] == posStr:
                TP += 1
            elif lPred[i] == negStr:
                TN += 1
            else:
                error_str = '%s and %s string is not in prediction, check posStr and negStr'%( posStr, negStr)
                raise( ValueError( error_str))
        elif lPred[i] == posStr:
            FP += 1
        elif lPred[i] == negStr:
            FN += 1
        else:
            error_str = '%s and %s string is not in prediction, check posStr and negStr'%( posStr, negStr)
            raise( ValueError( error_str))
    fAcc = (TP+TN)/( len( lPred)) #
    if TP+FP > 0:
        fPre = TP/(TP+FP) #What proportion of positive identifications is correct?
    else:
        fPre = 0
    if TN+FN == 0:
        fSpec = 0
    else:
        fSpec= TN/(TN+FN) #
    if TP+FP == 0:
        fSens = 0
    else:
        fSens= TP/(TP+FN)
    return { 'mConf' : np.array( [[TP, FN],[FP, TN]]), 'accuracy' : fAcc, 'precision': fPre, 'recall' : fSens }

def accuracy( a_y, a_y_hat):
    n_corr = 0.
    for i in range(len(a_y)):
        if a_y[i] == a_y_hat[i]:
            n_corr += 1
    return n_corr / (len(a_y_hat))
#---------------------------------4---------------------------------------
#                              plots
#-------------------------------------------------------------------------
def plot_confusion_matrix(ax, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    :input  ax -  axis instance
            cm - confusion matrix, nxn
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plot1 = ax.imshow(cm, interpolation='nearest', cmap=cmap, origin='lower')
    ax.set_title(title)
    plt.colorbar( plot1, shrink = .4)
    tick_marks = np.arange(len(classes))

    ax.set_xticks(tick_marks)
    ax.set_xticklabels( classes, rotation=25)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    #plt.tight_layout()
    
    
def plot_decision_regions( mX, a_y, Classifier, resolution = .02, test_idx=None):
    #
    from matplotlib.colors import ListedColormap
    # set-up markers and colors
    markers = ('s',    'x',      'o','^','v')
    colors  = ('red','blue', 'lightgreen','gray','cyan')
    # color for unique class labels - plot different regions
    cmap    = ListedColormap( colors[:len(np.unique(a_y))])
    # plot separation surfaces
    x1_min, x1_max = mX[:,0].min()-1, mX[:,0].max() + 1
    x2_min, x2_max = mX[:,1].min()-1, mX[:,1].max() + 1
    XX1, XX2 = np.meshgrid( np.arange( x1_min,x1_max,resolution),
                            np.arange( x2_min,x2_max,resolution))
    mZ = Classifier.predict( np.array([XX1.ravel(), XX2.ravel()]).T)
    mZ = mZ.reshape( XX1.shape)
    plt.contourf( XX1, XX2, mZ, alpha = .4, cmap = cmap)
    plt.xlim( x1_min, x1_max)
    plt.ylim( x2_min, x2_max)

    # plot input sample (instances)
    for idx,cl in enumerate( np.unique( a_y)):
        plt.scatter( x=mX[a_y==cl,0],y=mX[a_y==cl,1], alpha = .5, c=colors[idx],
                     marker = markers[idx], label=cl)
    # highlight test samples
    if test_idx:
        X_test, y_test = mX[test_idx,:], a_y[test_idx]
        plt.scatter( X_test[:,0], X_test[:,1], c='',alpha=1.0, linewidth=1,
                     marker='o',s=55,label = 'test data')
    
    
    
    