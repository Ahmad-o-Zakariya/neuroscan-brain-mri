import torch

def dice_score(logits,targets,eps=1e-7):
    p=(torch.sigmoid(logits)>.5).float(); dims=(1,2,3); inter=(p*targets).sum(dims); den=p.sum(dims)+targets.sum(dims)
    return torch.where(den==0,torch.ones_like(den),(2*inter+eps)/(den+eps)).mean()

def bce_dice_loss(logits,targets,bce_weight=.5):
    b=torch.nn.functional.binary_cross_entropy_with_logits(logits,targets); p=torch.sigmoid(logits); dims=(1,2,3); d=1-((2*(p*targets).sum(dims)+1e-6)/(p.sum(dims)+targets.sum(dims)+1e-6)).mean(); return bce_weight*b+(1-bce_weight)*d
