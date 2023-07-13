import argparse
from pathlib import Path
from datasets import get_data
import pandas as pd
import os
import numpy as np
import trainer
import tester
import torch
import matplotlib.pyplot as plt
import model
from sklearn.metrics import balanced_accuracy_score

def main(args):
	np.random.seed(seed=args.seed)
	torch.manual_seed(args.seed)
	device = args.device
	dataset = args.dataset
	data_dir = args.data_path
	supervised = True if args.supervised == 1 else False
	fn = args.data_path
	checkpoint_fn = 'checkpoints'
	isExist = os.path.exists(checkpoint_fn)
	if not isExist:
		# Create a new directory because it does not exist
		os.makedirs(data_dir)
		print("The new directory {} is created!".format(checkpoint_fn))

	transfer_evaluation = True if args.transfer_evaluation == 1 else False

	bac_inners_target_task = []
	inner_embeddings = {}

	# Get data
	train_data, test_data, target_task_train, target_task_test, sensitive_train, sensitive_test = get_data(dataset=dataset, data_dir=data_dir, seed=None, transfer_evaluation=False)
	data_shape = train_data.shape[1]

	if transfer_evaluation:
		idx_train_1, idx_train_0, idx_test_1, idx_test_0, train_data_1, train_data_0, test_data_1, test_data_0, target_task_train_1, target_task_train_0, target_task_test_1, target_task_test_0 = get_data(dataset=dataset, data_dir=data_dir, seed=None, transfer_evaluation=transfer_evaluation)

	# initialize inner and outer VAEs
	inner_latent_dim = args.inner_latent_dim
	outer_latent_dim = args.outer_latent_dim


	# regular training (not transfer evaluation)
	outer_VAE = model.outerVAE(in_size=data_shape,
	                           h_size=args.num_neurons,
	                           latent_size=outer_latent_dim,
	                           n_layers=args.n_layers,
	                           dropout_rate=args.dropout_rate,
	                           device=device).to(device)

	inner_VAE = model.innerVAE(in_size=outer_latent_dim,
	                           h_size=args.num_neurons,
	                           latent_size=inner_latent_dim,
	                           n_layers=args.n_layers,
	                           dropout_rate=args.dropout_rate,
	                           device=device).to(device)
	print('Train outer VAE first.')
	optimizer_outer = torch.optim.AdamW(outer_VAE.parameters(), lr=args.learning_rate_outer)
	if args.existing_outer_model_path == 'None':
		print('No existing model path specified, training from scratch for {} iterations.'.format(args.max_iters_outer))
		trainer.train(model=outer_VAE,
	              inner_model=False,
	              optimizer=optimizer_outer,
	              iterations=args.max_iters_outer,
	              device=device,
	              batch_size=args.batch_size,
	              save_iter=args.save_iter,
	              model_save_path=args.model_save_path,
	              eval_interval=args.eval_interval,
	              eval_iters=args.eval_iters,
	              train_data=train_data,
	              test_data=test_data,
	              target_task_train=target_task_train,
	              target_task_test=target_task_test,
		              kl_weight=args.kl_weight_outer,
			              sensitive_train=sensitive_train,
			              sensitive_test=sensitive_test, supervised=supervised
	              )

	else:  # if existing checkpoint file is given, compare iteration against max_iters and finish training if necessary
		print('Loading checkpoint file at: ', args.existing_model_path)
		checkpoint = torch.load(args.existing_outer_model_path)
		outer_VAE.load_state_dict(checkpoint['model_state_dict'])
		optimizer_outer.load_state_dict(checkpoint['optimizer_state_dict'])
		checkpoint_iter = checkpoint['iteration']

		if checkpoint_iter != args.max_iters_outer:
			trainer.train(model=outer_VAE,
			              inner_model=False,
			              optimizer=optimizer_outer,
			              iterations=args.max_iters_outer,
			              device=device,
			              batch_size=args.batch_size,
			              save_iter=args.save_iter,
			              model_save_path=args.model_save_path,
			              eval_interval=args.eval_interval,
			              eval_iters=args.eval_iters,
			              train_data=train_data,
			              test_data=test_data,
			              target_task_train=target_task_train,
			              target_task_test=target_task_test,
			              start_iter=checkpoint_iter,
		              kl_weight=args.kl_weight_outer,
			              sensitive_train=sensitive_train,
			              sensitive_test=sensitive_test, supervised=supervised
			              )


	print('Generate mean embeddings to train the inner VAE.')

	_, outer_embeddings_train, _ = outer_VAE(train_data.to(device))
	_, outer_embeddings_test, _ = outer_VAE(test_data.to(device))

	outer_embeddings_train = outer_embeddings_train.detach()
	outer_embeddings_test = outer_embeddings_test.detach()

	print('Train inner VAE.')
	optimizer_inner = torch.optim.AdamW(inner_VAE.parameters(), lr=args.learning_rate_inner)
	if args.existing_inner_model_path == 'None':  # if no specified checkpoint file is given, train the model
		print('No existing model path specified, training from scratch for {} iterations.'.format(args.max_iters_inner))
		trainer.train(model=inner_VAE,
		              inner_model=True,
		              optimizer=optimizer_inner,
		              iterations=args.max_iters_inner,
		              device=device,
		              batch_size=args.batch_size,
		              save_iter=args.save_iter,
		              model_save_path=args.model_save_path,
		              eval_interval=args.eval_interval,
		              eval_iters=args.eval_iters,
		              train_data=outer_embeddings_train,
		              test_data=outer_embeddings_test,
		              target_task_train=target_task_train,
		              target_task_test=target_task_test,
		              kl_weight=args.kl_weight_inner,
			              sensitive_train=sensitive_train,
			              sensitive_test=sensitive_test, supervised=supervised
		              )

	else:  # if existing checkpoint file is given, compare iteration against max_iters and finish training if necessary
		print('Loading inner checkpoint file at: ', args.existing_model_path)
		checkpoint = torch.load(args.existing_inner_model_path)
		inner_VAE.load_state_dict(checkpoint['model_state_dict'])
		optimizer_inner.load_state_dict(checkpoint['optimizer_state_dict'])
		checkpoint_iter = checkpoint['iteration']

		if checkpoint_iter != args.max_iters_inner:
			trainer.train(model=inner_VAE,
			              inner_model=True,
			              optimizer=optimizer_inner,
			              iterations=args.max_iters_inner,
			              device=device,
			              batch_size=args.batch_size,
			              save_iter=args.save_iter,
			              model_save_path=args.model_save_path,
			              eval_interval=args.eval_interval,
			              eval_iters=args.eval_iters,
			              train_data=outer_embeddings_train,
			              test_data=outer_embeddings_test,
			              target_task_train=target_task_train,
			              target_task_test=target_task_test,
			              start_iter=checkpoint_iter,
		              kl_weight=args.kl_weight_inner,
			              sensitive_train=sensitive_train,
			              sensitive_test=sensitive_test, supervised=supervised
			              )

	print('Generate mean embeddings from the inner VAE.')

	_, inner_embeddings_train, _ = inner_VAE(outer_embeddings_train.to(device))
	_, inner_embeddings_test, _ = inner_VAE(outer_embeddings_test.to(device))

	inner_embeddings_train = inner_embeddings_train.detach()
	inner_embeddings_test = inner_embeddings_test.detach()

	print('-------------RUNNING EVALUATIONS-------------')


	# Test whether the target factor can be predicted from the raw data
	BAC_raw_target_task = tester.predict_label(train_X=train_data, test_X=test_data, train_y=target_task_train, test_y=target_task_test)
	print('Balanced accuracy for target factor on original data (excluding sensitive factor):', BAC_raw_target_task)

	# Test whether the target factor can be predicted from the outerVAE embeddings
	BAC_outer_target_task = tester.predict_label(train_X=outer_embeddings_train, test_X=outer_embeddings_test, train_y=target_task_train,
	                                           test_y=target_task_test)
	print('Balanced accuracy for target factor on outerVAE embeddings:', BAC_outer_target_task)

	# Test whether the target factor can be predicted from the inner VAE embeddings
	BAC_inner_target_task = tester.predict_label(train_X=inner_embeddings_train, test_X=inner_embeddings_test, train_y=target_task_train,
	                                           test_y=target_task_test)
	print('Balanced accuracy for target factor on inner VAE embeddings:', BAC_inner_target_task)
	bac_inners_target_task.append(BAC_inner_target_task)

	# Test whether the sensitive factor can be predicted from the raw data
	BAC_raw_sensitive = tester.predict_label(train_X=train_data, test_X=test_data, train_y=sensitive_train, test_y=sensitive_test)
	print('Balanced accuracy for sensitive factor on original data (excluding sensitive factor):', BAC_raw_sensitive)

	# Test whether the sensitive factor can be predicted from the outer VAE embeddings
	BAC_outer_sensitive = tester.predict_label(train_X=outer_embeddings_train, test_X=outer_embeddings_test,
	                                           train_y=sensitive_train,
	                                           test_y=sensitive_test)

	print('Balanced accuracy for sensitive factor on outerVAE embeddings:', BAC_outer_sensitive)

	# Test whether the sensitive factor can be predicted from the inner VAE embeddings
	BAC_inner_sensitive = tester.predict_label(train_X=inner_embeddings_train, test_X=inner_embeddings_test,
	                                           train_y=sensitive_train,
	                                           test_y=sensitive_test)

	print('Balanced accuracy for sensitive factor on inner VAE embeddings:', BAC_inner_sensitive)

	if transfer_evaluation:
		# for each transfer domain, train and RDF on one domain and test it on the other

		inner_embeddings_train_1 = inner_embeddings_train[idx_train_1]
		inner_embeddings_train_0 = inner_embeddings_train[idx_train_0]

		inner_embeddings_test_1 = inner_embeddings_test[idx_test_1]
		inner_embeddings_test_0 = inner_embeddings_test[idx_test_0]

		print('---------------First Adjusted Parity  WITH strict transfer between domains...---------------')

		# Test whether the target factor can be predicted from the inner VAE embeddings
		BAC_inner_target_task_1 = tester.predict_label(train_X=inner_embeddings_train_1, test_X=inner_embeddings_test_1,
		                                             train_y=target_task_train_1,
		                                             test_y=target_task_test_1)

		BAC_inner_target_task_0 = tester.predict_label(train_X=inner_embeddings_train_0, test_X=inner_embeddings_test_0,
		                                               train_y=target_task_train_0,
		                                               test_y=target_task_test_0)

		bacs = np.array([BAC_inner_target_task_1, BAC_inner_target_task_0])

		old_min = 0.5
		old_max = 1.0
		adj_parity_score = tester.adjusted_parity_two_domains(accs=bacs, rescale_acc_min=old_min, rescale_acc_max=old_max)

		print('Balanced Accuracies on the two domains:', bacs)
		print('Average Balanced Accuracy over the two domains:', bacs.mean())
		print('Adjusted Parity Metric:', adj_parity_score)

		print('---------------Second Adjusted Parity WITHOUT strict transfer between domains...---------------')

		predictors_train = torch.cat([inner_embeddings_train_1, inner_embeddings_train_0], 0)
		predictors_test = torch.cat([inner_embeddings_test_1, inner_embeddings_test_0], 0)

		outcome_train = torch.cat([target_task_train_1, target_task_train_0])
		outcome_test = torch.cat([target_task_test_1, target_task_test_0])

		preds = tester.predict_label(train_X=predictors_train, test_X=predictors_test,
		                                               train_y=outcome_train,
		                                               test_y=outcome_test,
		                                               return_preds=True)


		preds_1, preds_0 = preds[:len(target_task_test_1)], preds[len(target_task_test_1):]

		bac1 = balanced_accuracy_score(target_task_test_1, preds_1)
		bac0 = balanced_accuracy_score(target_task_test_0, preds_0)
		bacs_no_transfer = np.array([bac1, bac0])
		old_min = 0.5
		old_max = 1.0
		adj_parity_score_no_transfer = tester.adjusted_parity_two_domains(accs=bacs_no_transfer, rescale_acc_min=old_min, rescale_acc_max=old_max)

		print('Balanced Accuracies on the two domains WITHOUT transfer:', bacs_no_transfer)
		print('Average Balanced Accuracy over the two domains WITHOUT transfer:', bacs_no_transfer.mean())
		print('Adjusted Parity Metric WITHOUT transfer:', adj_parity_score_no_transfer)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--dataset",
		type=str,
		default='synth1',
		required=True,
		help="Name of the dataset to be used for training and/or testing."
	)
	parser.add_argument(
		"--device",
		type=str,
		default='cuda',
		help="Device to train and/or run the model ('cuda' or 'cpu')."
	)

	parser.add_argument(
		"--existing_inner_model_path",
		type=str,
		required=False,
		default='None',
		help="Path to a model checkpoint file (if not specified, will use random init).",
	)

	parser.add_argument(
		"--existing_outer_model_path",
		type=str,
		required=False,
		default='None',
		help="Path to a model checkpoint file (if not specified, will use random init).",
	)

	parser.add_argument(
		"--model_save_path",
		type=Path,
		required=False,
		default='checkpoints/',
		help="Path to a model checkpoint filename.",
	)

	parser.add_argument(
		"--data_path",
		type=Path,
		required=False,
		default='data',
		help="Path to data.",
	)

	parser.add_argument(
		"--batch_size",
		type=int,
		default=10,
		help="Batch size for training"
	)
	parser.add_argument(
		"--max_iters_inner",
		type=int,
		default=100,
		help="Iterations for training the inner VAE."
	)
	parser.add_argument(
		"--max_iters_outer",
		type=int,
		default=100,
		help="Iterations for training the outer VAE."
	)
	parser.add_argument(
		"--eval_interval",
		type=int,
		default=10,
		help="Number of training iterations which pass before evaluation."
	)
	parser.add_argument(
		"--eval_iters",
		type=int,
		default=10,
		help="Number of evaluation batches to estimate loss with."
	)
	parser.add_argument(
		"--learning_rate_inner",
		type=float,
		default=3e-4,
		help="Iterations for training the inner VAE."
	)
	parser.add_argument(
		"--learning_rate_outer",
		type=float,
		default=3e-4,
		help="Iterations for training the outer VAE."
	)
	parser.add_argument(
		"--save_iter",
		type=int,
		default=100,
		help="Iterations for model checkpointing."
	)

	parser.add_argument(
		"--num_neurons",
		type=int,
		default=4,
		help="Embedding dimension within the transformer."
	)

	parser.add_argument(
		"--dropout_rate",
		type=float,
		default=0.3,
		help="Dropout probability during training."
	)
	parser.add_argument(
		"--n_layers",
		type=int,
		default=8,
		help="Number of layers in enc <and> dec. Should be at least 3."
	)

	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed."
	)

	parser.add_argument(
		"--validation_fraction",
		type=float,
		default=0.3,
		help="Percentage of data used for validation"
	)

	parser.add_argument(
		"--inner_latent_dim",
		type=int,
		default=6,
		help="Dimensionality of the latent space for the inner VAE."
	)

	parser.add_argument(
		"--outer_latent_dim",
		type=int,
		default=6,
		help="Dimensionality of the latent space for the outer VAE."
	)
	parser.add_argument(
		"--kl_weight_outer",
		type=float,
		default=1.0,
		help="The Beta in BetaVAE (the associated weight on the KL part of the loss)."
	)

	parser.add_argument(
		"--kl_weight_inner",
		type=float,
		default=1.0,
		help="The Beta in BetaVAE (the associated weight on the KL part of the loss)."
	)
	parser.add_argument(
		"--supervised",
		type=int,
		default=1,
		help="1 if you want to use the sensitive factor itself during training (as supervision), 0 otherwise."
	)

	parser.add_argument(
		"--transfer_evaluation",
		type=int,
		default=1,
		help="1 if you want to train on data from one class and test on another (to evaluate transfer learning), 0 if you want to train and test on all data (normal)"
	)
	args = parser.parse_args()

	main(args)

