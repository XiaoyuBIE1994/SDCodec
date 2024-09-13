import torch
from torch.optim import lr_scheduler

def warmup_learning_rate(optimizer, nb_iter, warmup_iter, max_lr):
    """warmup learning rate"""
    lr = max_lr * nb_iter / warmup_iter
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_scheduler(optimizer, args):
    if args.policy == 'linear':
        scheduler = lr_scheduler.LinearLR(optimizer, total_iters=args.total_iter) # factor 0.33-1
    elif args.policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_iter)
    elif args.policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=args.decay_step, gamma=0.1)
    elif args.policy == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=0.05)
    elif args.policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate args.policy [%s] is not implemented', args.policy)
    return scheduler


def Configure_AdamW(model, weight_decay, learning_rate):

    all_params = set(model.parameters())
    decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)

    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)):
            decay.add(m.weight)
    no_decay = all_params - decay

    # create the pytorch optimizer object
    optim_groups = [
        {"params": list(decay), "weight_decay": weight_decay},
        {"params": list(no_decay), "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer