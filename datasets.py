

import torch
import numpy as np
import os

def get_data(dataset, data_dir, seed=None):

	if dataset == 'adult':

		train_data = np.load(os.path.join(data_dir, 'Adult_train_data_no_gender.npy')).astype(np.float32)
		test_data = np.load(os.path.join(data_dir, 'Adult_test_data_no_gender.npy')).astype(np.float32)

		target_task_train = torch.from_numpy(train_data[:, -1]).to(torch.bool)
		target_task_test = torch.from_numpy(test_data[:, -1]).to(torch.bool)

		train_data = train_data[:, :-1]
		test_data = test_data[:, :-1]

		sensitive_train = torch.load(os.path.join(data_dir, 'ADULT_GENDER_TRAIN.pt')).to(torch.bool)
		sensitive_test = torch.load(os.path.join(data_dir, 'ADULT_GENDER_TEST.pt')).to(torch.bool)

		train_data_mu = train_data.mean(0)
		train_data_std = train_data.std(0)
		train_data = (train_data - train_data_mu) / train_data_std
		test_data = (test_data - train_data_mu) / train_data_std

		train_data = torch.from_numpy(train_data)
		test_data = torch.from_numpy(test_data)

	else:
		print('Sorry, dataset does not yet exist. Try using "adult".')

	return train_data, test_data, target_task_train, target_task_test, sensitive_train, sensitive_test


