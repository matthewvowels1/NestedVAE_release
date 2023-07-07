# NestedVAE_release
Release for Nested VAE


python3 main.py --dataset adult \
--max_iters 40000 \
--validation_fraction 0.3 \
--device cuda \
--existing_inner_model_path  None \
--existing_outer_model_path  None \
--model_save_path checkpoints \
--data_path data \
--batch_size 32 \
--num_neurons 32 \
--eval_interval 2000 \
--eval_iters 50 \
--learning_rate 1e-3 \
--save_iter 10000 \
--dropout_rate 0.3 \
--n_layers 5 \
--inner_latent_dim 6 \
--outer_latent_dim 6  \
--kl_weight 0.0 \
--supervised 1
