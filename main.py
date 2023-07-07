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
	if transfer_evaluation:
		assert not supervised,  'Cannot be supervised if evaluating for transfer learning.'

	bac_inners_target_task = []
	inner_embeddings = {}

	# Get data
	if transfer_evaluation:
		train_data_1, train_data_0, test_data_1, test_data_0, target_task_train_1, target_task_train_0, target_task_val_1, target_task_val_0 = get_data(dataset=dataset, data_dir=data_dir, seed=None, transfer_evaluation=transfer_evaluation)
		data_shape = train_data_1.shape[1]
	else:
		train_data, test_data, target_task_train, target_task_val, sensitive_train, sensitive_val = get_data(dataset=dataset, data_dir=data_dir, seed=None, transfer_evaluation=transfer_evaluation)
		data_shape = train_data.shape[1]

	# initialize inner and outer VAEs
	inner_latent_dim = args.inner_latent_dim
	outer_latent_dim = args.outer_latent_dim

	bac_transfer = {}
	if transfer_evaluation:
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
		              train_data=train_data_1,
		              val_data=test_data_1,
		              target_task_train=target_task_train_1,
		              target_task_val=target_task_val_1,
			              kl_weight=args.kl_weight_outer,
				              sensitive_train=None,
				              sensitive_val=None, supervised=supervised
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
				              train_data=train_data_1,
				              val_data=test_data_1,
				              target_task_train=target_task_train_1,
				              target_task_val=target_task_val_1,
				              start_iter=checkpoint_iter,
			              kl_weight=args.kl_weight_outer,
				              sensitive_train=None,
				              sensitive_val=None, supervised=supervised
				              )


		print('Generate mean embeddings to train and test the inner VAE.')

		_, outer_embeddings_train_1, _ = outer_VAE(train_data_1.to(device))
		_, outer_embeddings_test_1, _ = outer_VAE(test_data_1.to(device))

		_, outer_embeddings_train_0, _ = outer_VAE(train_data_0.to(device))
		_, outer_embeddings_test_0, _ = outer_VAE(test_data_0.to(device))

		outer_embeddings_train_1 = outer_embeddings_train_1.detach()
		outer_embeddings_test_1 = outer_embeddings_test_1.detach()

		outer_embeddings_train_0 = outer_embeddings_train_0.detach()
		outer_embeddings_test_0 = outer_embeddings_test_0.detach()

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
			              train_data=outer_embeddings_train_1,
			              val_data=outer_embeddings_test_1,
			              target_task_train=target_task_train_1,
			              target_task_val=target_task_val_1,
			              kl_weight=args.kl_weight_inner,
				              sensitive_train=None,
				              sensitive_val=None, supervised=supervised
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
				              train_data=outer_embeddings_train_1,
				              val_data=outer_embeddings_test_1,
				              target_task_train=target_task_train_1,
				              target_task_val=target_task_val_1,
				              start_iter=checkpoint_iter,
			              kl_weight=args.kl_weight_inner,
				              sensitive_train=None,
				              sensitive_val=None, supervised=supervised
				              )

		print('Generate mean embeddings from the inner VAE.')

		_, inner_embeddings_train_1, _ = inner_VAE(outer_embeddings_train_1.to(device))
		_, inner_embeddings_test_1, _ = inner_VAE(outer_embeddings_test_1.to(device))

		_, inner_embeddings_train_0, _ = inner_VAE(outer_embeddings_train_0.to(device))
		_, inner_embeddings_test_0, _ = inner_VAE(outer_embeddings_test_0.to(device))

		inner_embeddings_train_1 = inner_embeddings_train_1.detach()
		inner_embeddings_test_1 = inner_embeddings_test_1.detach()

		inner_embeddings_train_0 = inner_embeddings_train_0.detach()
		inner_embeddings_test_0 = inner_embeddings_test_0.detach()


		# Test whether the target factor can be predicted from the inner VAE embeddings in the transfer domain
		BAC_inner_target_task = tester.predict_label(train_X=inner_embeddings_train_0, test_X=inner_embeddings_test_0, train_y=target_task_train_0,
		                                           test_y=target_task_val_0)
		print('Balanced accuracy for target factor on inner VAE embeddings:', BAC_inner_target_task)
		bac_transfer['train_1_test_0'] = BAC_inner_target_task

		######################## now repeat for other transfer domain ########################
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
			print('No existing model path specified, training from scratch for {} iterations.'.format(
				args.max_iters_outer))
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
			              train_data=train_data_0,
			              val_data=test_data_0,
			              target_task_train=target_task_train_0,
			              target_task_val=target_task_val_0,
			              kl_weight=args.kl_weight_outer,
			              sensitive_train=None,
			              sensitive_val=None, supervised=supervised
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
				              train_data=train_data_0,
				              val_data=test_data_0,
				              target_task_train=target_task_train_0,
				              target_task_val=target_task_val_0,
				              start_iter=checkpoint_iter,
				              kl_weight=args.kl_weight_outer,
				              sensitive_train=None,
				              sensitive_val=None, supervised=supervised
				              )

		print('Generate mean embeddings to train and test the inner VAE.')

		_, outer_embeddings_train_0, _ = outer_VAE(train_data_0.to(device))
		_, outer_embeddings_test_0, _ = outer_VAE(test_data_0.to(device))

		_, outer_embeddings_train_1, _ = outer_VAE(train_data_1.to(device))
		_, outer_embeddings_test_1, _ = outer_VAE(test_data_1.to(device))

		outer_embeddings_train_0 = outer_embeddings_train_0.detach()
		outer_embeddings_test_0 = outer_embeddings_test_0.detach()

		outer_embeddings_train_1 = outer_embeddings_train_1.detach()
		outer_embeddings_test_1 = outer_embeddings_test_1.detach()

		print('Train inner VAE.')
		optimizer_inner = torch.optim.AdamW(inner_VAE.parameters(), lr=args.learning_rate_inner)
		if args.existing_inner_model_path == 'None':  # if no specified checkpoint file is given, train the model
			print('No existing model path specified, training from scratch for {} iterations.'.format(
				args.max_iters_inner))
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
			              train_data=outer_embeddings_train_0,
			              val_data=outer_embeddings_test_0,
			              target_task_train=target_task_train_0,
			              target_task_val=target_task_val_0,
			              kl_weight=args.kl_weight_inner,
			              sensitive_train=None,
			              sensitive_val=None, supervised=supervised
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
				              train_data=outer_embeddings_train_0,
				              val_data=outer_embeddings_test_0,
				              target_task_train=target_task_train_0,
				              target_task_val=target_task_val_0,
				              start_iter=checkpoint_iter,
				              kl_weight=args.kl_weight_inner,
				              sensitive_train=None,
				              sensitive_val=None, supervised=supervised
				              )

		print('Generate mean embeddings from the inner VAE.')

		_, inner_embeddings_train_0, _ = inner_VAE(outer_embeddings_train_0.to(device))
		_, inner_embeddings_test_0, _ = inner_VAE(outer_embeddings_test_0.to(device))

		_, inner_embeddings_train_1, _ = inner_VAE(outer_embeddings_train_1.to(device))
		_, inner_embeddings_test_1, _ = inner_VAE(outer_embeddings_test_1.to(device))

		inner_embeddings_train_0 = inner_embeddings_train_0.detach()
		inner_embeddings_test_0 = inner_embeddings_test_0.detach()

		inner_embeddings_train_1 = inner_embeddings_train_1.detach()
		inner_embeddings_test_1 = inner_embeddings_test_1.detach()

		# Test whether the target factor can be predicted from the inner VAE embeddings in the transfer domain
		BAC_inner_target_task = tester.predict_label(train_X=inner_embeddings_train_1, test_X=inner_embeddings_test_1,
		                                             train_y=target_task_train_1,
		                                             test_y=target_task_val_1)
		print('Balanced accuracy for target factor on inner VAE embeddings:', BAC_inner_target_task)
		bac_transfer['train_0_test_1'] = BAC_inner_target_task

		# compute adjusted parity across the transfer domains
		bacs = np.array([bac_transfer['train_0_test_1'], bac_transfer['train_1_test_0']])
		bac_norm = (bacs - 0.5) / (1.0 - np.min(bacs))
		av_bac = bac_norm.mean()
		std_bac = bac_norm.std()
		adj_parity = av_bac * (1 - 2 * std_bac)
		print('Adjusted Parity across the two transfer domains: ', adj_parity, ' for ', bacs )

	elif not transfer_evaluation:
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
		              val_data=test_data,
		              target_task_train=target_task_train,
		              target_task_val=target_task_val,
			              kl_weight=args.kl_weight_outer,
				              sensitive_train=sensitive_train,
				              sensitive_val=sensitive_val, supervised=supervised
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
				              val_data=test_data,
				              target_task_train=target_task_train,
				              target_task_val=target_task_val,
				              start_iter=checkpoint_iter,
			              kl_weight=args.kl_weight_outer,
				              sensitive_train=sensitive_train,
				              sensitive_val=sensitive_val, supervised=supervised
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
			              val_data=outer_embeddings_test,
			              target_task_train=target_task_train,
			              target_task_val=target_task_val,
			              kl_weight=args.kl_weight_inner,
				              sensitive_train=sensitive_train,
				              sensitive_val=sensitive_val, supervised=supervised
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
				              val_data=outer_embeddings_test,
				              target_task_train=target_task_train,
				              target_task_val=target_task_val,
				              start_iter=checkpoint_iter,
			              kl_weight=args.kl_weight_inner,
				              sensitive_train=sensitive_train,
				              sensitive_val=sensitive_val, supervised=supervised
				              )

		print('Generate mean embeddings from the inner VAE.')

		_, inner_embeddings_train, _ = inner_VAE(outer_embeddings_train.to(device))
		_, inner_embeddings_test, _ = inner_VAE(outer_embeddings_test.to(device))

		inner_embeddings_train = inner_embeddings_train.detach()
		inner_embeddings_test = inner_embeddings_test.detach()


		# Test whether the target factor can be predicted from the raw data
		BAC_raw_target_task = tester.predict_label(train_X=train_data, test_X=test_data, train_y=target_task_train, test_y=target_task_val)
		print('Balanced accuracy for target factor on original data (excluding sensitive factor):', BAC_raw_target_task)

		# Test whether the target factor can be predicted from the outerVAE embeddings
		BAC_outer_target_task = tester.predict_label(train_X=outer_embeddings_train, test_X=outer_embeddings_test, train_y=target_task_train,
		                                           test_y=target_task_val)
		print('Balanced accuracy for target factor on outerVAE embeddings:', BAC_outer_target_task)

		# Test whether the target factor can be predicted from the inner VAE embeddings
		BAC_inner_target_task = tester.predict_label(train_X=inner_embeddings_train, test_X=inner_embeddings_test, train_y=target_task_train,
		                                           test_y=target_task_val)
		print('Balanced accuracy for target factor on inner VAE embeddings:', BAC_inner_target_task)
		bac_inners_target_task.append(BAC_inner_target_task)

		# Test whether the sensitive factor can be predicted from the raw data
		BAC_raw_sensitive = tester.predict_label(train_X=train_data, test_X=test_data, train_y=sensitive_train, test_y=sensitive_val)
		print('Balanced accuracy for sensitive factor on original data (excluding sensitive factor):', BAC_raw_sensitive)

		# Test whether the sensitive factor can be predicted from the outer VAE embeddings
		BAC_outer_sensitive = tester.predict_label(train_X=outer_embeddings_train, test_X=outer_embeddings_test,
		                                           train_y=sensitive_train,
		                                           test_y=sensitive_val)

		print('Balanced accuracy for sensitive factor on outerVAE embeddings:', BAC_outer_sensitive)

		# Test whether the sensitive factor can be predicted from the inner VAE embeddings
		BAC_inner_sensitive = tester.predict_label(train_X=inner_embeddings_train, test_X=inner_embeddings_test,
		                                           train_y=sensitive_train,
		                                           test_y=sensitive_val)

		print('Balanced accuracy for sensitive factor on inner VAE embeddings:', BAC_inner_sensitive)





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

