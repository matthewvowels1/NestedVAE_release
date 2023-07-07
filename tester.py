

import imblearn
from sklearn.metrics import balanced_accuracy_score


def predict_label(train_X, test_X, train_y, test_y):
	'''
	A function for checking how easily the sensitive factor (y) can be predicted using a balanced RDF

	:param train_X: a set of D-dimensional input predictors (N, D), torch
	:param test_X: a set of D-dimensional input predictors (N, D), torch
	:param train_y: a set of 1D labels for the sensitive factor, torch
	:param test_y:  a set of 1D labels for the sensitive factor, torch
	:return: Balanced accuracy score
	'''

	train_X = train_X.cpu().detach().numpy()
	test_X = test_X.cpu().detach().numpy()
	train_y = train_y.cpu().detach().numpy()
	test_y = test_y.cpu().detach().numpy()

	# first train RDF to check baseline 'bias' in the original data
	rf_clf = imblearn.ensemble.BalancedRandomForestClassifier(warm_start=False, n_estimators=100)
	rf_clf.fit(train_X, train_y)

	test_preds = rf_clf.predict(test_X)

	BAC = balanced_accuracy_score(test_y, test_preds)
	return BAC
