# ----Escort criterion
def escort_function(x, p=2):
    x_exp = torch.abs(x)**p
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    answer = x_exp/x_exp_sum

    return answer

def escort_criterion(device):
    criterion = nn.KLDivLoss()
    if device is not None:
        criterion = criterion.cuda()
    return lambda pred, true : criterion((escort_function(pred)+ 1e-7).log(), nn.functional.one_hot(true, num_classes=pred.shape[1]).float())
#------


