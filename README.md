# NestedVAE_release
Release for Nested VAE: https://openaccess.thecvf.com/content_CVPR_2020/papers/Vowels_NestedVAE_Isolating_Common_Factors_via_Weak_Supervision_CVPR_2020_paper.pdf

Uses the Adult dataset.

Example CLI:

python3 main.py --dataset adult \
--max_iters_inner 100000 \
--max_iters_outer 50000 \
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
--learning_rate_outer 1e-3 \
--learning_rate_inner 1e-4 \
--save_iter 50000 \
--dropout_rate 0.3 \
--n_layers 5 \
--inner_latent_dim 6 \
--outer_latent_dim 6  \
--kl_weight 0.0 \
--supervised 1
