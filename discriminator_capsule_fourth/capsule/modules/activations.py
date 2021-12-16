import torch


def squash(inputs, axis=-1):
    """
    Do squashing
    to reduce the length of capsules
    :param inputs (tensor): tensor of dim [M, num_capsules, capsule_dim]
    :param axis: the dim to apply squash
    :return:
        (tensor): squashed tensor of dim [M, num_capsules, capsule_dim]
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)  # compute the length of input caps,p=2 is second fanshu
    # keepdim=true is to keep the original space shape
    scale = norm ** 2 / (1 + norm ** 2) / (norm + 1e-8)  # compute suo fang yin zi
    return scale * inputs   # finish the computation

