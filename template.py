#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/uvebeen/oss_prj2.git

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as mt

def load_dataset(dataset_path):
	#To-Do: Implement this function
	dataframe = pd.read_csv(dataset_path)

	return dataframe

def dataset_stat(dataset_df):
	#To-Do: Implement this function
	number_of_features = dataset_df.shape[1] - 1
	number_of_class0 = len(dataset_df.loc[dataset_df['target'] == 0])
	number_of_class1 = len(dataset_df.loc[dataset_df['target'] == 1])

	return number_of_features, number_of_class0, number_of_class1

def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
	data = dataset_df.drop(columns="target", axis=1)
	label = dataset_df["target"]

	data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=testset_size)

	return data_train, data_test, label_train, label_test

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	dt_cls = DecisionTreeClassifier()
	dt_cls.fit(x_train, y_train)

	y_pred = dt_cls.predict(x_test)

	dt_accuracy = mt.accuracy_score(y_test, y_pred)
	dt_precision = mt.precision_score(y_test, y_pred)
	dt_recall = mt.recall_score(y_test, y_pred)

	return dt_accuracy, dt_precision, dt_recall

def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	rf_cls = RandomForestClassifier()
	rf_cls.fit(x_train, y_train)

	y_pred = rf_cls.predict(x_test)

	rf_accuracy = mt.accuracy_score(y_test, y_pred)
	rf_precision = mt.precision_score(y_test, y_pred)
	rf_recall = mt.recall_score(y_test, y_pred)

	return rf_accuracy, rf_precision, rf_recall

def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	svm_pipe = make_pipeline(
		StandardScaler(),
		SVC()
	)
	svm_pipe.fit(x_train, y_train)

	y_pred = svm_pipe.predict(x_test)

	svm_accuracy = mt.accuracy_score(y_test, y_pred)
	svm_precision = mt.precision_score(y_test, y_pred)
	svm_recall = mt.recall_score(y_test, y_pred)

	return svm_accuracy, svm_precision, svm_recall

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)