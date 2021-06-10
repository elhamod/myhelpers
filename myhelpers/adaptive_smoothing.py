import torch

def get_lambdas(adaptive_alpha, adaptive_lambda, fine_loss, other_losses):
    c = -(1-adaptive_alpha)/adaptive_lambda
    terms = {
        'fine': c*fine_loss
    }
    for loss_name in other_losses:
        terms[loss_name] = c*other_losses[loss_name]

    p_values = {}
    for loss_name in terms:
        p=0
        try:
            for term in terms:
                p = p + torch.exp(terms[term] - terms[loss_name])
            p = 1/p
        except:
            p=0
        p_values[loss_name] = p
    # print('p_values', p_values)

        

    lambdas = {
        'fine': (adaptive_alpha + (1-adaptive_alpha)*p_values['fine']).detach().item()
    }
    for loss_name in other_losses:
        lambdas[loss_name] = ((1-adaptive_alpha)*p_values[loss_name]).detach().item()
    # print('lambdas', lambdas)

    return lambdas

# Two loss functions
def get_total_adaptive_loss(adaptive_alpha, adaptive_lambda, loss_fine, loss_coarse):
    c = -(1-adaptive_alpha)/adaptive_lambda
    c_fine = c*loss_fine
    c_coarse = c*loss_coarse
    
    c_diff = c_coarse - c_fine
    try:
        p = 1/(1+ torch.exp(c_diff))
    except:
        p=0

    # Not stable (Nans)
    # n_fine = torch.exp(c_fine)
    # print(n_fine)
    # n_coarse = torch.exp(c_coarse)
    # print(n_coarse)
    # p = n_fine/ (n_fine + n_coarse)
    # print(p)

    lambda_fine = adaptive_alpha + (1-adaptive_alpha)*p
    lambda_coarse = (1-adaptive_alpha)*(1-p)
    return lambda_fine.detach().item(), lambda_coarse.detach().item()
