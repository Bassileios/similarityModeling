from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import random
import numpy as np


db = ['Muppets-03-04-03-final.csv','Muppets-02-01-01-final.csv','Muppets-02-04-04-final.csv']


# Loads the Dataset specified by the Number
def load_data(num):
	data = open(db[num], 'r')
	line = data.readline().strip().split()
	res = []

	while len(line) > 1:
		l = [0]*(len(line)-1)
		for i in range(1,len(line)):
			l[i-1] = float(line[i])
		res.append(l)
		line = data.readline().strip().split()
	return res


# Splits the Data into Samples and Lables
def split_data(data):
	n = len(data)
	m = len(data[0])
	X = [0]*n
	y = [0]*n
	for i in range(n):
		X[i] = data[i][0:m-1]
		y[i] = int(data[i][m-1])
	return X,y


# Saves the Results for Accuracy, Precision and Recall
def safe_results(file_name, acc, prec, rec):
	with open(file_name, 'w') as file:
		file.write('Accuracy\tPrecision\tRecall\n')
		for i in range(len(acc)):
			file.write('{}\t{}\t{}\n'.format(acc[i], prec[i], rec[i]))
	return


# Main Procedure, Trains three Models and Creates a Statistic
def main():
	print('Loading in Datasets')
	statistic = ['SVM.csv', 'kNN.csv', 'DecTree.csv']
	data = load_data(0)
	data += load_data(1)
	data += load_data(2)

	validate = 150
	acc_SVM = [0.0]*validate
	rec_SVM = [0.0]*validate
	prec_SVM = [0.0]*validate

	acc_kNN = [0.0]*validate
	rec_kNN = [0.0]*validate
	prec_kNN = [0.0]*validate

	acc_DT = [0.0]*validate
	rec_DT = [0.0]*validate
	prec_DT = [0.0]*validate

	for i in range(validate):
		print('Iteration', i)
		random.shuffle(data)
		X,y = split_data(data)
		partition = int(0.3*len(X))

		print('Classify with SVM')
		support = svm.LinearSVC(max_iter=1000)
		result = cross_validate(support, X[0:partition], y[0:partition], cv=10, scoring=['accuracy', 'recall', 'precision'])
		acc_SVM[i] = np.mean(result['test_accuracy'])
		prec_SVM[i] = np.mean(result['test_precision'])
		rec_SVM[i] = np.mean(result['test_recall'])
		print('\tAccuracy:\t',acc_SVM[i])
		print('\tPrecision:\t',prec_SVM[i])
		print('\tRecall:\t',rec_SVM[i])


		print('Classify with kNN')
		nbrs = KNeighborsClassifier(n_neighbors=5)
		result = cross_validate(nbrs, X[0:partition], y[0:partition], cv=10, scoring=['accuracy', 'recall', 'precision'])
		acc_kNN[i] = np.mean(result['test_accuracy'])
		prec_kNN[i] = np.mean(result['test_precision'])
		rec_kNN[i] = np.mean(result['test_recall'])
		print('\tAccuracy:\t',acc_kNN[i])
		print('\tPrecision:\t',prec_kNN[i])
		print('\tRecall:\t',rec_kNN[i])


		print('Classify with DT')
		dt = DecisionTreeClassifier(random_state=0)
		result = cross_validate(dt, X[0:partition], y[0:partition], cv=10, scoring=['accuracy', 'recall', 'precision'])
		acc_DT[i] = np.mean(result['test_accuracy'])
		prec_DT[i] = np.mean(result['test_precision'])
		rec_DT[i] = np.mean(result['test_recall'])
		print('\tAccuracy:\t',acc_DT[i])
		print('\tPrecision:\t',prec_DT[i])
		print('\tRecall:\t',rec_DT[i])


	print('')
	print('')
	print('Results from', validate, 'Random Samples of 30% of overall Data:')
	print('\tSVM:')
	print('\tAccuracy:\t', np.mean(acc_SVM))
	print('\tPrecision:\t', np.mean(prec_SVM))
	print('\tRecall:\t', np.mean(rec_SVM))
	print('')
	print('\tkNN:')
	print('\tAccuracy:\t', np.mean(acc_kNN))
	print('\tPrecision:\t', np.mean(prec_kNN))
	print('\tRecall:\t', np.mean(rec_kNN))
	print('')
	print('\tDecision Tree:')
	print('\tAccuracy:\t', np.mean(acc_DT))
	print('\tPrecision:\t', np.mean(prec_DT))
	print('\tRecall:\t', np.mean(rec_DT))

	print('Saving Statistic')
	safe_results(statistic[0], acc_SVM, prec_SVM, rec_SVM)
	safe_results(statistic[1], acc_kNN, prec_kNN, rec_kNN)
	safe_results(statistic[2], acc_DT, prec_DT, rec_DT)
	print('Finished')
	return


main()




