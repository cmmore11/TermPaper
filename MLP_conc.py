#!~opt/anaconda3/bin/python3.7
"""
Data Analysis of Concrete Mix Dataset
sURL = "http://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength"
    y: target classifier, concrete mix category ("Comp_Range" in Excel)
    n: There are 8 possible attributes to help predict the target classifier
Code Returns:
        Correlation Matrix
        Summary of input
        Training and Testing Data Set Sizes
        Training and Testing Accuracy of MLP Classifier
        Confusion Matrix
Author:Christie Moore Assadollahi
"""
#import Modules
import matplotlib as mpl
mpl.use('Qt5Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as skMetric
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier as SL_MLP
import ml_utils as ml_utils
#-------data-import---------------
data_file = 'Concrete.xlsx'
all_atts = ['Cement','Slag','FlyAsh','Water','SuperP',
            'Coarse','Fine','Age','W/C','CompStr','Numb']
# Test different attributes to use based off of correlation matrix
l_use_attr= ['Cement','Slag','FlyAsh','Water','Age'] 

#load data - panda data frame
o_pdf = pd.read_excel(data_file)
# select specific class labels
label = 'Numb'
#=================================1============================================
#              Hyperparameters for SL_MLP Classifier
#==============================================================================
max_iter           = 300 # Maximum Number of iterations
hidden_layer_sizes = 140 # number of hidden units
alpha              = 0.0001 #set to 0 to switch regularization off  
learning_rate_init = 0.015#learning rate
validation_fraction= 0.10
test_size    = 0.20 # fraction of samples in testing data set
#=================================2============================================
#              Data Visualization and Cleaning
#==============================================================================
#Histogram of compressive strengths
plt.figure(1)
plt.hist(o_pdf['CompStr'],10,edgecolor='k')
plt.title('Compressive Strength Distribution')
plt.xlabel('Compressive Strength')
plt.ylabel('Frequency')
plt.show()
plt.savefig('Compressive Strengths Histogram')

#Histogram of classified strengths:uniform
plt.figure(2)
plt.hist(o_pdf['Numb'],5,edgecolor='k')
plt.title('Compressive Strength Categories')
plt.xlabel('Compressive Strength Category')
plt.ylabel('Frequency')
plt.show()
plt.savefig('Classified Strengths Histogram')

# Create and print correlation matrix:
cor = o_pdf.corr()
cor.style.background_gradient(cmap='coolwarm').set_precision(2)
# adding labels source: https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/
f = plt.figure(3,figsize=(8,6))
ax = f.add_subplot(111)
cax = ax.matshow(cor,cmap = plt.cm.coolwarm_r, vmin = -1, vmax = 1)
cb = f.colorbar(cax)
cb.ax.tick_params(labelsize=14)
ticks = np.arange(0,11,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(all_atts,rotation = 45)
ax.set_yticklabels(all_atts,rotation = 45)
f.show()
plt.savefig('Correlation Matrix')

print('Summary of Data')
print('     Min. Compressive Strength: %.1f MPa'%(np.min(o_pdf['CompStr'])))
print('     Max. Compressive Strength: %.1f MPa'%(np.max(o_pdf['CompStr'])))
print('     Avg. Compressive Strength: %.1f MPa'%(np.average(o_pdf['CompStr'])))
print('     c, unique class labels: %d'%((len(np.unique(o_pdf[label])))))
print('     n, instances          : %d'%(len(o_pdf)))
print('     m, attributes         : %d'%(len(all_atts)))

#=================================3============================================
#              Data Preprocessing, Test/Train Split
#==============================================================================
use_class= np.sort(np.unique(o_pdf[label])) #[0,1,2,3,4], detected 
#Attributes and Classifiers        
a_y     = o_pdf[label].values               #if classifier is a string: a_y_int = ml_utils.str2int(a_y)
m_X = o_pdf[l_use_attr].values              #if attribute is a string:m_X[:,0] = ml_utils.str2int(m_X[:,0])

X_train, X_test, y_train, y_test = train_test_split( m_X, a_y, shuffle=True,
                                                     test_size=test_size)
print('Test-Train Split')
print('     Training: Rows: %d, Cols: %d' % (X_train.shape[0], X_train.shape[1]))
print('     Testing : Rows: %d, Cols: %d'%(X_test.shape[0], X_test.shape[1]))
#=================================4============================================
#              Training and Prediction
#==============================================================================
o_nn = SL_MLP(hidden_layer_sizes = hidden_layer_sizes, activation = 'logistic',
              solver = 'sgd', alpha = alpha, learning_rate = 'adaptive', 
              learning_rate_init = learning_rate_init, max_iter = max_iter, 
              shuffle = True , warm_start=True, 
              early_stopping = True, validation_fraction = validation_fraction)

o_nn.fit( X_train, y_train) #three steps of perceptron

y_predictTRAIN = o_nn.predict( X_train)
y_predict = o_nn.predict( X_test)
#=================================5============================================
#              evaluate performance
#==============================================================================
#Zero Rule Prediction (Baseline)
SK_0R = DummyClassifier(strategy="most_frequent")#most_frequent,'stratified
y_predict_0Rtrain = SK_0R.fit( X_train, y_train).predict( y_train)
print('Performance of MLP')
print('-----Training Accuracy-----')
print( 'baseline accuracy: %.2f'%(skMetric.accuracy_score( y_train, y_predict_0Rtrain)))
print( 'MLP accuracy     : %.2f'%(skMetric.accuracy_score( y_train, y_predictTRAIN)))
print('-----Testing Accuracy-----')
y_predict_0R = SK_0R.fit( X_train, y_train).predict( y_test)
print( 'baseline accuracy: %.2f'%(skMetric.accuracy_score( y_test, y_predict_0R)))
print( 'MLP accuracy     : %.2f'%(skMetric.accuracy_score( y_test, y_predict,normalize=True)))

#=================================6============================================
#             Cost Function and Confusion Matrix
#==============================================================================
#--evaluate performance at every epoch/iteration---------------------------
plt.figure(4)
plt.plot( o_nn.loss_curve_, 'ko-', label = ['Cost Function'])
plt.xlabel( 'Iterations')
plt.ylabel( 'Cost')
plt.legend()
plt.savefig( 'LossCurve')
plt.show()
#------------------- Confusion Matrix-----------------------
# confusion matrix
classLABEL = ['0%-20%','20%-40%','40%-60%','60%-80%','80%-100%']
m_CM = skMetric.confusion_matrix( y_test,y_predict)
plt.figure(5, figsize=(8,6))
ax3 = plt.subplot( 111)
ml_utils.plot_confusion_matrix(ax3, m_CM, classLABEL)
plt.show()
plt.savefig('ConfusionMatrix')



