import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False):
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction=reduction, zero_infinity=zero_infinity)
