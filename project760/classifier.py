import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import sys



data_dir = './input/'	# needs trailing slash

# validation split, both files with headers and the Happy column
train_file = data_dir + 'census.data.v5.csv'
test_file = data_dir + 'census.test.v5.csv'

###
train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

# train = large_train.sample(int(sys.argv[1]))
# test = large_test.sample(int(sys.argv[1]))
# y
y_train = train.Label
y_test = test.Label
print type(y_test)
# populate lists of numeric and cat attributes

all_cols = list(train.columns.values)
numeric_cols = ['age','wage per hour','capital gains','capital losses','dividends from stocks','num persons worked for employer','weeks worked in year',]
# remove_cols = ['Label','instance weight','migration code-change in msa','migration code-change in reg','migration code-move within reg','migration prev res in sunbelt']
remove_cols = ['Label','instance weight']
cat_cols =  [x for x in all_cols if x not in numeric_cols and x not in remove_cols]

# handle numerical features
x_num_train = train[ numeric_cols ].as_matrix()
x_num_test = test[ numeric_cols ].as_matrix()

x_train_count = x_num_train.shape[0]
x_test_count = x_num_test.shape[0]

x_num_combined = np.concatenate((x_num_train,x_num_test), axis=0) # 0 -row 1 - col

# scale numeric features to <0,1>
max_num = np.amax( x_num_combined, 0 )

x_num_combined = np.true_divide(x_num_combined, max_num) # scale by max. truedivide needed for decimals
x_num_train = x_num_combined[0:x_train_count]
x_num_test = x_num_combined[x_train_count:]

# categorical

x_cat_train = train.drop( numeric_cols + remove_cols, axis = 1 )
x_cat_test = test.drop( numeric_cols + remove_cols, axis = 1 )

x_cat_train.fillna( 'NA', inplace = True )
x_cat_test.fillna( 'NA', inplace = True )

x_cat_combined = pd.concat((x_cat_train, x_cat_test), axis=0)

# one-of-k handling for categorical features
# pandas.get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False)
vec_x_cat_combined = pd.get_dummies(x_cat_combined, columns=cat_cols, sparse=False)#.as_matrix()

vec_x_cat_train = vec_x_cat_combined[0:x_train_count]
vec_x_cat_test = vec_x_cat_combined[x_train_count:]

# combine numerical and categorical

x_train = np.hstack(( x_num_train, vec_x_cat_train )) # returns ndarray
x_test = np.hstack(( x_num_test, vec_x_cat_test ))

# working fine upto here - Data Processing
# below - classifier specific logic

naive_bayes = GaussianNB()
naive_bayes.fit(x_train,y_train,sample_weight=train['instance weight'].values)

predicted = naive_bayes.predict( x_test )

probs = naive_bayes.predict_proba(x_test)
y_conf=[]
y_test_num=[]
print "Before for reset"
print len(y_test)
# print y_test
for i in range(len(y_test)):
	if y_test[i] == 'Pos':
		y_test_num.append(1)
	else:
		y_test_num.append(0)

#Neg is 0 in probs and 0 in y_test
#pos is 1 in probs and 1 in y_test
print "Before Plot"

for class_to_plot in [0,1]:
	y_conf = []
	for i in range(len(y_test)):
		y_conf.append(probs[i][class_to_plot])
	precision, recall, thresholds = precision_recall_curve(y_test_num, y_conf, pos_label=class_to_plot, sample_weight=test['instance weight'].values)
	print "after PR ",class_to_plot
	plt.plot(recall,precision)
	if(class_to_plot == 0):
		plt.axis([0,1,0.8,1])
	else:
		plt.axis([0,1,0,1])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('NB with instance weights for class ' + str(class_to_plot))
	filename = "./plots/nb_weighted_"+str(class_to_plot)+".png"
	plt.savefig(filename)
	plt.clf()


#Learn without Instance weights

naive_bayes = GaussianNB()
naive_bayes.fit( x_train, y_train )

predicted = naive_bayes.predict( x_test )

for class_to_plot in [0,1]:
	y_conf = []
	for i in range(len(y_test)):
		y_conf.append(probs[i][class_to_plot])
	precision, recall, thresholds = precision_recall_curve(y_test_num, y_conf, pos_label=class_to_plot)
	plt.plot(recall,precision)
	plt.axis([0,1,0,1])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('NB without instance weights for class ' + str(class_to_plot))
	filename = "./plots/nb_unweighted_"+str(class_to_plot)+".png"
	plt.savefig(filename)
	plt.clf()


