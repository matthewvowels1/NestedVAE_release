# NestedVAE_release
Release for Nested VAE: https://openaccess.thecvf.com/content_CVPR_2020/papers/Vowels_NestedVAE_Isolating_Common_Factors_via_Weak_Supervision_CVPR_2020_paper.pdf

Runs the Adult dataset.

-------------RUNNING EVALUATIONS-------------
Balanced accuracy for target factor on original data (excluding sensitive factor): 0.8358033135734915

Balanced accuracy for target factor on outerVAE embeddings: 0.7255028626437655

Balanced accuracy for target factor on inner VAE embeddings: 0.7133809596189897

Balanced accuracy for sensitive factor on original data (excluding sensitive factor): 0.8512705672986699

Balanced accuracy for sensitive factor on outerVAE embeddings: 0.6613172533304057

Balanced accuracy for sensitive factor on inner VAE embeddings: 0.6315625354451732

---------------First Adjusted Parity  WITH strict transfer between domains...---------------

Balanced Accuracies on the two domains: [0.66696958 0.7418032 ]

Average Balanced Accuracy over the two domains: 0.704386387770519

Adjusted Parity Metric: 0.34759288642904607

---------------Second Adjusted Parity WITHOUT strict transfer between domains...---------------

Balanced Accuracies on the two domains WITHOUT transfer: [0.67653127 0.73383648]

Average Balanced Accuracy over the two domains WITHOUT transfer: 0.7051838763083335

Adjusted Parity Metric WITHOUT transfer: 0.3633353328614705




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
