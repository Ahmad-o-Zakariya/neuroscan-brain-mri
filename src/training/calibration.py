"""Validation-only operating threshold selection."""
import numpy as np

def select_tumour_threshold(tumour_probabilities, tumour_labels, target_sensitivity=.95):
    p=np.asarray(tumour_probabilities,dtype=float).reshape(-1); y=np.asarray(tumour_labels,dtype=bool).reshape(-1)
    if len(p)!=len(y) or not len(p): raise ValueError("Calibration requires matching non-empty validation probabilities and labels.")
    if not y.any(): raise ValueError("Tumour gate calibration requires at least one tumour-positive validation case.")
    if not 0<target_sensitivity<=1: raise ValueError("target_sensitivity must be in (0, 1].")
    positives=np.sort(p[y]); index=max(0,int(np.floor((1-target_sensitivity)*len(positives))))
    threshold=float(positives[min(index,len(positives)-1)])
    sensitivity=float(np.mean(p[y]>=threshold)); specificity=float(np.mean(p[~y]<threshold)) if (~y).any() else None
    return {"threshold":threshold,"target_sensitivity":float(target_sensitivity),"validation_sensitivity":sensitivity,"validation_specificity":specificity,"n_validation":len(y),"n_tumour_positive":int(y.sum()),"method":"largest empirical threshold at the lower positive-probability quantile meeting target sensitivity"}

def select_mask_threshold(probability_maps, masks, candidates=None):
    probs=np.asarray(probability_maps,dtype=float); truth=np.asarray(masks,dtype=bool)
    if probs.shape!=truth.shape or probs.ndim<2 or probs.shape[0]==0: raise ValueError("Mask calibration requires matching non-empty probability maps and masks.")
    candidates=np.asarray(candidates if candidates is not None else np.linspace(.1,.9,17),dtype=float)
    if not len(candidates) or np.any((candidates<0)|(candidates>1)): raise ValueError("Mask threshold candidates must lie in [0,1].")
    scores=[]
    for threshold in candidates:
        pred=probs>=threshold; axes=tuple(range(1,probs.ndim)); inter=np.logical_and(pred,truth).sum(axis=axes); den=pred.sum(axis=axes)+truth.sum(axis=axes)
        dice=np.where(den==0,1.,(2*inter)/np.maximum(den,1));scores.append(float(dice.mean()))
    best=max(range(len(scores)),key=lambda i:(scores[i],-abs(candidates[i]-.5)))
    return {"threshold":float(candidates[best]),"validation_dice_per_image_mean":scores[best],"n_validation":int(probs.shape[0]),"method":"maximizes mean per-image Dice over configured threshold grid"}
