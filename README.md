# NestedVAE_release
Release for Nested VAE: https://openaccess.thecvf.com/content_CVPR_2020/papers/Vowels_NestedVAE_Isolating_Common_Factors_via_Weak_Supervision_CVPR_2020_paper.pdf

Runs the Adult dataset.

When transfer_evaluation = False:
Balanced accuracy for sensitive factor on original data (excluding sensitive factor): 0.8516335258720195
Balanced accuracy for target factor on original data (excluding sensitive factor): 0.8329596696559761
Balanced accuracy for sensitive factor on outerVAE embeddings: 0.6594728609008579
Balanced accuracy for target factor on outerVAE embeddings: 0.7278588437959164
Balanced accuracy for sensitive factor on inner VAE embeddings: 0.6339402052438536
Balanced accuracy for target factor on inner VAE embeddings: 0.70786467041597

When transfer_evaluation = True:
Balanced accuracy for target factor on inner VAE embeddings: 0.5962623839569002
Adjusted Parity across the two transfer domains:  0.2614595002631862  for  domain scores [0.59626238 0.76223673]


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
--kl_weight_inner 0.0 \
--kl_weight_outer 0.0 \
--transfer_evaluation 1 \
--supervised 0
