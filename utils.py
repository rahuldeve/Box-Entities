from typing import *
import random
import numpy as np
import torch
import torch.nn as nn

# from box_embeddings.parameterizations.sigmoid_box_tensor import SigmoidBoxTensor
from box_embeddings.parameterizations import BoxTensor

from tqdm import tqdm

from model import DualEncoder

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def get_all_mention_embeds(loader, model, device):
#     """
#     get all the mentions embeddings
#     :return:
#     """
#     model.eval()
#     all_men_embeds = []
#     if hasattr(model, 'module'):
#         encoder = model.module.mention_encoder
#     else:
#         encoder = model.mention_encoder
#     with torch.no_grad():
#         for i, batch in enumerate(loader):
#             men_embeds = encoder(
#                 input_ids=batch[0].to(device), 
#                 attention_mask=batch[1].to(device)
#             )
#             men_embeds = men_embeds[0][:, 0, :].detach().cpu()
#             all_men_embeds.append(men_embeds)

#     all_men_embeds = torch.cat(all_men_embeds, dim=0)
#     return all_men_embeds


def unwrap_dataparallel_model(wrapped_dual_encoder) -> DualEncoder:
    if hasattr(wrapped_dual_encoder, 'module'):
        encoder = wrapped_dual_encoder.module
    else:
        encoder = wrapped_dual_encoder

    return encoder


def get_all_entity_embeds(
    en_loader, 
    model, 
    device, 
    dp
):
    model.eval()
    model: DualEncoder = unwrap_dataparallel_model(model)
    entity_base_encoder = model.entity_encoder.encoder
    if dp:
        entity_base_encoder = nn.DataParallel(entity_base_encoder)

    all_en_embeds = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(en_loader), total=len(en_loader)):
            en_embeds = entity_base_encoder(
                input_ids=batch[0].to(device),
                attention_mask=batch[1].to(device)
            )
            en_embeds = en_embeds[0][:, 0, :]
            en_embeds = en_embeds.detach().cpu()
            all_en_embeds.append(en_embeds)

    all_en_embeds = torch.cat(all_en_embeds, dim=0)
    return all_en_embeds
    




class IntersectionalVolumeRatio(nn.Module):
    def __init__(self, all_en_embeds, containment_func, box_factory):
        super().__init__()
        self.containment_func = containment_func
        self.box_fatory = box_factory
        self.all_en_embeds = nn.parameter.Parameter(all_en_embeds)

    def forward(self, men_embeds):
        all_en_boxes: BoxTensor = self.box_fatory.from_vector(self.all_en_embeds)
        men_boxes: BoxTensor = self.box_fatory.from_vector(men_embeds)

        n_mens = men_boxes.box_shape[0]
        n_cands = self.all_en_embeds.shape[0]
        logits = torch.zeros((n_mens, n_cands), device=self.all_en_embeds.device)

        for i in range(n_mens):
            men_box = men_boxes[[i]]
            int_log_prob = self.containment_func(men_box, all_en_boxes)
            # int_log_prob = self.box_vol(self.box_int(men_box, all_en_boxes)) - self.box_vol(all_en_boxes)
            logits[i, :] = torch.exp(int_log_prob)

        return logits


def get_hard_negative(men_loader, model, all_en_embeds, num_hards, device, dp):
    model.eval()
    model: DualEncoder = unwrap_dataparallel_model(model)
    mention_base_encoder = model.mention_encoder.encoder
    if dp:
        mention_base_encoder = nn.DataParallel(mention_base_encoder)


    scorer = IntersectionalVolumeRatio(all_en_embeds, model.containment_function, model.box_factory)
    scorer = scorer.to(device)
    scorer = nn.DataParallel(scorer)

    hard_indices = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(men_loader), total=len(men_loader)):
            men_embeds = mention_base_encoder(
                input_ids=batch[0].to(device),
                attention_mask=batch[1].to(device)
            )
            men_embeds = men_embeds[0][:, 0, :]

            logits = scorer(men_embeds)
            _, hard_cands = logits.topk(num_hards + 1, dim=1)

            labels = batch[2].to(device)
            mask = (hard_cands != labels)
            for j in range(hard_cands.size(0)):
                hard_cands[j][:num_hards] = hard_cands[j][mask[j]][:num_hards]

            hard_indices.append(hard_cands[:, :num_hards])
        
    hard_indices = torch.cat(hard_indices, dim=0).long().cpu().tolist()
    return hard_indices


def evaluate(en_loader, men_loader, model, k, device):
    model.eval()

    with torch.no_grad():
        all_en_embeds = get_all_entity_embeds(en_loader, model, device, dp=True)


    nb_samples = 0
    r_k = 0
    acc = 0

    model: DualEncoder = unwrap_dataparallel_model(model)
    mention_base_encoder = model.mention_encoder.encoder
    mention_base_encoder = nn.DataParallel(mention_base_encoder)
        
    scorer = IntersectionalVolumeRatio(all_en_embeds, model.containment_function, model.box_factory)
    scorer = scorer.to(device)
    scorer = nn.DataParallel(scorer)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(men_loader), total=len(men_loader)):
            men_embeds = mention_base_encoder(
                input_ids=batch[0].to(device),
                attention_mask=batch[1].to(device)
            )
            men_embeds = men_embeds[0][:, 0, :]
            logits = scorer(men_embeds)
            _, top_k = logits.topk(k, dim=1)
            
            labels = batch[2].to(device)
            preds = top_k[:, 0]
            r_k += (top_k == labels.to(device)).sum().item()
            nb_samples += men_embeds.size(0)
            acc += (preds == labels.squeeze(1).to(device)).sum().item()
            
    r_k /= nb_samples
    acc /= nb_samples
    return r_k, acc