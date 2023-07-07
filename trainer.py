import torch
import numpy as np
import os

recon_func = torch.nn.MSELoss()

def vae_loss(y_true, y_pred, log_sigma, mu):
    """ Compute loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    recon = recon_func(y_pred, y_true)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * torch.sum(torch.exp(log_sigma) + mu ** 2 - 1 - log_sigma, dim=1)
    kl = torch.sum(kl, dim=0)
    return recon, kl


def get_inner_batch(train_data, val_data, split, batch_size, target_task_train, target_task_val, device, sensitive_train=None, sensitive_val=None, supervised=False):
    '''
    Takes data, target task labels and (if supervised=True) sensitive factor labels and creates vae
     input/reconstruction pairings according to these labels.
    :param train_data:
    :param val_data:
    :param split:
    :param batch_size:
    :param target_task_train:
    :param target_task_val:
    :param device:
    :param sensitive_train:
    :param sensitive_val:
    :param supervised: True/False
    :return:
    '''

    data = train_data if split == 'train' else val_data
    target_task = target_task_train if split == 'train' else target_task_val

    if supervised:
        sensitive = sensitive_train if split == 'train' else sensitive_val

        # get indices according to values of target and sensitive values
        indices_T1_S1 = (target_task & sensitive).nonzero(as_tuple=True)[0]
        indices_T1_S0 = (target_task & ~sensitive).nonzero(as_tuple=True)[0]
        indices_T0_S1 = (~target_task & sensitive).nonzero(as_tuple=True)[0]
        indices_T0_S0 = (~target_task & ~sensitive).nonzero(as_tuple=True)[0]

        # get data subsets according to these criteria
        data_T1_S1 = data[indices_T1_S1]
        data_T1_S0 = data[indices_T1_S0]
        data_T0_S1 = data[indices_T0_S1]
        data_T0_S0 = data[indices_T0_S0]

        # put a batch together by sampling from those subsets of the data
        ix_T1_S1 = torch.randint(0, len(data_T1_S1), (batch_size//4,))
        ix_T1_S0 = torch.randint(0, len(data_T1_S0), (batch_size // 4,))
        ix_T0_S1 = torch.randint(0, len(data_T0_S1), (batch_size // 4,))
        ix_T0_S0 = torch.randint(0, len(data_T0_S0), (batch_size // 4,))

        quarter_batch_T1_S1 = data_T1_S1[ix_T1_S1]
        quarter_batch_T1_S0 = data_T1_S1[ix_T1_S0]
        quarter_batch_T0_S1 = data_T1_S1[ix_T0_S1]
        quarter_batch_T0_S0 = data_T1_S1[ix_T0_S0]

        x_in = torch.cat([quarter_batch_T1_S1, quarter_batch_T1_S0, quarter_batch_T0_S1, quarter_batch_T0_S0])
        x_out = torch.cat([quarter_batch_T1_S0, quarter_batch_T1_S1, quarter_batch_T0_S0, quarter_batch_T0_S1])

    else:
        pos_indices = (target_task == 1).nonzero(as_tuple=True)[0]
        neg_indices = (target_task == 0).nonzero(as_tuple=True)[0]
        pos_class_data = data[pos_indices]
        neg_class_data = data[neg_indices]
        ix_pos1 = torch.randint(0, len(pos_class_data), (batch_size//2,))  # get half a batch for the positive target class
        ix_neg1 = torch.randint(0, len(neg_class_data), (batch_size//2,))  # get half a batch for the negative target class
        ix_pos2 = torch.randint(0, len(pos_class_data), (batch_size // 2,))  # get second half of a batch for the positive target class
        ix_neg2 = torch.randint(0, len(neg_class_data), (batch_size // 2,))  # get second half of a batch for the negative target class
        x_pos1 = pos_class_data[ix_pos1]
        x_neg1 = neg_class_data[ix_neg1]
        x_pos2 = pos_class_data[ix_pos2]
        x_neg2 = neg_class_data[ix_neg2]
        x_in = torch.cat([x_pos1, x_neg1])  # compile the full INPUT batch of randomly selected positive (first) and negative (second) target class
        x_out = torch.cat([x_pos2, x_neg2])  # compile the full TARGET batch (for reconstruction) of randomly selected positive (first) and negative (second) target class

    return x_in.to(device), x_out.to(device)


def get_outer_batch(train_data, val_data, split, batch_size, device):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data), (batch_size,))
    x = data[ix]
    return x.to(device)

@torch.no_grad()
def estimate_loss(model, model_name, train_data, val_data, batch_size, eval_iters, device, kl_weight, target_task_train=None, target_task_val=None, sensitive_train=None, sensitive_val=None, supervised=False):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):

            if model_name == 'inner':
                xb, target = get_inner_batch(train_data=train_data,
                                     val_data=val_data,
                                     split=split,
                                     device=device,
                                     target_task_train=target_task_train,
                                     target_task_val=target_task_val,
                                     batch_size=batch_size,
                                             sensitive_train=sensitive_train,
                                             sensitive_val=sensitive_val,supervised=supervised)

            elif model_name == 'outer':
                xb = get_outer_batch(train_data=train_data,
                                     val_data=val_data,
                                     split=split,
                                     device=device,
                                     batch_size=batch_size)
                target = xb

            pred, mu, log_sigma = model(xb)
            recon_loss, kl_loss = vae_loss(y_true=target, y_pred=pred, mu=mu, log_sigma=log_sigma)
            loss = recon_loss + (kl_weight * kl_loss)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()
    return out

def train(model, inner_model, optimizer, train_data, val_data, device, batch_size, save_iter, model_save_path, eval_interval, eval_iters, iterations=1000, start_iter=None, kl_weight=1.0, supervised=False, target_task_train=None, target_task_val=None, sensitive_train=None, sensitive_val=None):

    model_name = 'inner' if inner_model else 'outer'

    if start_iter == None:
        start_iter = 0

    for iter_ in range(start_iter, iterations):
        if model_name == 'inner':
            xb, target = get_inner_batch(train_data=train_data,
                                         val_data=val_data,
                                         target_task_train=target_task_train,
                                         target_task_val=target_task_val,
                                         split='train',
                                         device=device,
                                         batch_size=batch_size,
                                             sensitive_train=sensitive_train,
                                             sensitive_val=sensitive_val,
                                         supervised=supervised)
        elif model_name == 'outer':
            xb = get_outer_batch(train_data=train_data, val_data=val_data, split='train', device=device, batch_size=batch_size)
            target = xb

        if iter_ % eval_interval == 0:
            print('Evaluating model.')
            losses = estimate_loss(model=model,
                                   model_name=model_name,
                                        train_data=train_data,
                                        val_data=val_data,
                                        eval_iters=eval_iters,
                                        batch_size=batch_size,
                                        device=device,
                                       target_task_train=target_task_train,
                                       target_task_val=target_task_val,
                                   sensitive_train=sensitive_train,
                                   sensitive_val=sensitive_val,
                                   kl_weight=kl_weight,
                                   supervised=supervised)


            print(f"step {iter_}: train_loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        pred, mu, log_sigma = model(xb)
        recon_loss, kl_loss = vae_loss(y_true=target, y_pred=pred, mu=mu, log_sigma=log_sigma)
        loss = recon_loss + (kl_weight * kl_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iter_ > 1) and (iter_ != start_iter) and ((iter_ + 1) % save_iter == 0):
            print('Saving model checkpoint.')
            torch.save({
                'iteration': iter_,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(model_save_path, '{}_model_{}_{}.ckpt'.format(model_name, iter_ + 1, np.round(loss.item(), 2))))

    return model