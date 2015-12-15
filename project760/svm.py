import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn import svm
from sklearn import metrics
from sklearn import neighbors, datasets
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


data_dir = './input/'	# needs trailing slash

# validation split, both files with headers and the Happy column
train_file = data_dir + 'census.data.v3.csv'
test_file = data_dir + 'census.test.v3.csv'

###

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

# print train['instance weight'].values

all_cols = list(train.columns.values)
# numeric x
numeric_cols = ['age','wage per hour','capital gains','capital losses','dividends from stocks','num persons worked for employer','weeks worked in year',]

remove_cols = ['Label','instance weight','migration code-change in msa','migration code-change in reg','migration code-move within reg','migration prev res in sunbelt']
#l3 = [x for x in l1 if x not in l2]
cat_cols =  [x for x in all_cols if x not in numeric_cols and x not in remove_cols]


x_num_train = train[ numeric_cols ].as_matrix()
x_num_test = test[ numeric_cols ].as_matrix()


# scale to <0,1>

max_train = np.amax( x_num_train, 0 )
max_test = np.amax( x_num_test, 0 )		# not really needed

x_num_train = np.true_divide(x_num_train, max_train)
x_num_test = np.true_divide(x_num_test, max_train)		# scale test by max_train


# y
y_train = train.Label
y_test = test.Label

# categorical

cat_train = train.drop( numeric_cols + ['Label','instance weight','migration code-change in msa','migration code-change in reg','migration code-move within reg','migration prev res in sunbelt'], axis = 1 )
cat_test = test.drop( numeric_cols + ['Label','instance weight','migration code-change in msa','migration code-change in reg','migration code-move within reg','migration prev res in sunbelt'], axis = 1 )

cat_train.fillna( 'NA', inplace = True )
cat_test.fillna( 'NA', inplace = True )

print("TRAIN CAT dims: ", cat_train.shape)
print("TEST CAT dims: ", cat_test.shape)



#   pandas.get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False)
vec_x_cat_train = pd.get_dummies(cat_train, columns=cat_cols, sparse=False).as_matrix()
vec_x_cat_test = pd.get_dummies(cat_test, columns=cat_cols, sparse=False).as_matrix()

#print(type(vec_x_cat_test))

print("TRAIN CAT dims: ", vec_x_cat_train.shape)
print("TRAIN NUM dims: ", x_num_train.shape)

print("TEST CAT dims: ", vec_x_cat_test.shape)
print("TEST NUM dims: ", x_num_test.shape)

# problem : some of the cateogrical feature vals do not appear in the test file.. vice versa also possible.
# todo : ndarray.concatenate - combine train and test
# Then split into num vs cat, do all processing, do hstack.
# finally do vsplit to split into train and test again before training.

# complete 

x_train = np.hstack(( x_num_train, vec_x_cat_train )) # returns ndarray
x_test = np.hstack(( x_num_test, vec_x_cat_test ))

svm_classifier = svm.SVC(probability=True)
svm_classifier.fit( x_train, y_train , sample_weight=train['instance weight'].values)

predicted = svm_classifier.predict( x_test )
expected = y_test

print(metrics.classification_report(expected, predicted))

probs = svm_classifier.predict_proba(x_test)
y_conf=[]
for i in range(len(y_test)):
	if y_test[i] == 'Pos':
		y_test[i] = 1
	else:
		y_test[i] = -1

for i in range(len(expected)):
	y_conf.append(probs[i][1])

precision, recall, thresholds = precision_recall_curve(expected, y_conf, pos_label=-1, sample_weight=test['instance weight'].values)
print precision
print recall
print thresholds
plt.plot(recall,precision)
plt.axis([0,1,0,1])
plt.show()

