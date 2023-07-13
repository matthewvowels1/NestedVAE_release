

import imblearn
from sklearn.metrics import balanced_accuracy_score


def adjusted_parity_two_domains(accs, rescale_acc_min=0.5, rescale_acc_max=1.0):
	'''Takes in an array of TWO accuracy scores (or balanced accuracy, or whatever) across two sensitive domains,
	 returns the adjusted parity metric score.

	 :param accs: is an array (length 2) of accuracy scores
	 :param rescale_acc_max: is the maximum possible accuracy, which will be used to rescale to 0.0-1.0
	 :param rescale_acc_min:  is the chance level accuracy, which will be used to rescale to 0.0-1.0
	 :return adjusted parity score.
	 '''

	new_max = 1.0
	new_min = 0.0

	accs_rescaled = ((accs - rescale_acc_min) / (rescale_acc_max - rescale_acc_min)) * (new_max - new_min) + new_min

	adj_par = accs_rescaled.mean() * (1 - (2 * accs_rescaled.std()))

	return adj_par




def predict_label(train_X, test_X, train_y, test_y, return_preds=False):
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

	if return_preds:
		return test_preds
	else:
		return BAC
