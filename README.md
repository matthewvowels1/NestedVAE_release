# NestedVAE_release
Release for Nested VAE: https://openaccess.thecvf.com/content_CVPR_2020/papers/Vowels_NestedVAE_Isolating_Common_Factors_via_Weak_Supervision_CVPR_2020_paper.pdf

Runs the Adult dataset.

Balanced accuracy for target factor on original data (excluding sensitive factor): 0.8358033135734915

Balanced accuracy for target factor on outerVAE embeddings: 0.7255028626437655

Balanced accuracy for target factor on inner VAE embeddings: 0.7252368647717484

Balanced accuracy for sensitive factor on original data (excluding sensitive factor): 0.8512705672986699

Balanced accuracy for sensitive factor on outerVAE embeddings: 0.6613172533304057

Balanced accuracy for sensitive factor on inner VAE embeddings: 0.6547469995566131

Balanced Accuracies on the two domains: [0.67235903 0.74341637]

Average Balanced Accuracy over the two domains: 0.7078876999995822

Deviation betwen domains: 0.07105733840479778

Adjusted Parity Metric: 0.3566876134029028


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
--supervised 1  # this was used in experiments but did not prove to help much (it uses the sensitive label as a way to design the random pairings for inner VAE).
