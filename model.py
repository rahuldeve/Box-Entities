from turtle import forward
from transformers import BertModel, BertPreTrainedModel, BertConfig
import torch
import torch.nn as nn
import copy
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from box_embeddings.parameterizations import BoxTensor
from box_embeddings.parameterizations.delta_box_tensor import MinDeltaBoxTensor
from box_embeddings.parameterizations.sigmoid_box_tensor import SigmoidBoxTensor
from box_embeddings.modules.volume import Volume
from box_embeddings.modules.intersection import Intersection
from box_embeddings.modules.regularization import l2_side_regularizer


class MentionBoxEncoder(nn.Module):
    def __init__(self, encoder, box_factory) -> None:
        super().__init__()
        self.encoder = encoder
        self.box_factory = box_factory

    def forward(
        self, 
        mention_token_ids: torch.Tensor,        # (B x d)
        mention_masks: torch.Tensor             # (B x d)
    ) -> BoxTensor:

        # mention_token_ids = mention_token_ids.long()
        # mention_masks = mention_masks.long()

        lm_output = self.encoder(input_ids=mention_token_ids, attention_mask=mention_masks)[0]
        box_data = lm_output[:, 0, :].unsqueeze(1)
        mention_box = self.box_factory.from_vector(box_data)
        return mention_box




class EntityBoxEncoder(nn.Module):
    def __init__(self, encoder, box_factory) -> None:
        super().__init__()
        self.encoder = encoder
        self.box_factory = box_factory

    def forward(
        self, 
        entity_candidate_token_ids: torch.Tensor,      # (B X C X L)
        entity_candidate_masks: torch.Tensor           # (B X C X L)
    ) -> BoxTensor:

        # entity_candidate_token_ids = entity_candidate_token_ids.long()
        # entity_candidate_masks = entity_candidate_masks.long()

        # B x d --> B x 1 x d
        B, C, L = entity_candidate_token_ids.size()
        entity_candidate_token_ids = entity_candidate_token_ids.view(-1, L)
        entity_candidate_masks = entity_candidate_masks.view(-1, L)
        # B X C X L --> BC X L
        lm_output = self.encoder(
            input_ids=entity_candidate_token_ids,
            attention_mask=entity_candidate_masks
        )[0]
        
        box_data = lm_output[:, 0, :]
        # BC X d --> B X C X d --> B X C X d
        box_data = box_data.view(B, C, -1)
        entity_candidates_boxes = self.box_factory.from_vector(box_data)
        return entity_candidates_boxes

        


class DualEncoder(nn.Module):
    def __init__(self, encoder, vol_temp, vol_int_temp, int_temp) -> None:
        super(DualEncoder, self).__init__()
        self.box_factory = MinDeltaBoxTensor
        self.mention_encoder = MentionBoxEncoder(encoder, self.box_factory)
        self.entity_encoder = EntityBoxEncoder(copy.deepcopy(encoder), self.box_factory)
        self.loss_fct = CrossEntropyLoss()
        self.box_volume = Volume(volume_temperature=vol_temp, intersection_temperature=vol_int_temp)
        self.box_intersection = Intersection(intersection_temperature=int_temp)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def containment_function(self, mention_box, entity_box) -> torch.Tensor:
        intersection_volumes = (
            self.box_volume(self.box_intersection(mention_box, entity_box)) - self.box_volume(mention_box)
        )
        return intersection_volumes
        
    def forward(
        self, 
        mention_token_ids: torch.Tensor, 
        mention_masks: torch.Tensor,
        candidate_token_ids: torch.Tensor,
        candidate_masks: torch.Tensor,
    ):
        mention_token_ids = mention_token_ids.to(self.device).long()
        mention_masks = mention_masks.to(self.device).long()
        candidate_token_ids = candidate_token_ids.to(self.device).long()
        candidate_masks = candidate_masks.to(self.device).long()

        mention_boxes = self.mention_encoder(mention_token_ids, mention_masks)
        entity_candidates_boxes = self.entity_encoder(candidate_token_ids, candidate_masks)
        logits = self.containment_function(mention_boxes, entity_candidates_boxes)

        B = mention_token_ids.size(0)
        labels = torch.zeros(B).long().to(self.device)
        loss = self.loss_fct(logits, labels)
        # loss += l2_side_regularizer(candidates_boxes).mean() + l2_side_regularizer(mention_boxes).mean()
        return loss, logits
