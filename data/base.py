import os
import random
import json
import torch
from torch.utils.data import Dataset, DataLoader

from data.entity import EntitySet
from data.mention import MentionSet

def load_data(data_dir):
    """

    :param data_dir: train_data_dir if args.train else eval_data_dir
    :return: mentions, entities,doc
    """
    print('begin loading data')
    men_path = os.path.join(data_dir, 'mentions')

    def load_mentions(part):
        mentions = []
        domains = set()
        with open(os.path.join(men_path, '%s.json' % part)) as f:
            for line in f:
                field = json.loads(line)
                mentions.append(field)
                domains.add(field['corpus'])
        return mentions, domains

    samples_train, train_domain = load_mentions('train')
    samples_heldout_train_seen, heldout_train_domain = load_mentions('heldout_train_seen')
    samples_heldout_train_unseen, heldout_train_unseen_domain = load_mentions('heldout_train_unseen')
    samples_val, val_domain = load_mentions('val')
    samples_test, test_domain = load_mentions('test')

    def load_entities(data_dir, domains):
        """

        :param domains: list of domains
        :return: all the entities in the domains
        """
        doc = {}
        doc_path = os.path.join(data_dir, 'documents')
        for domain in domains:
            with open(os.path.join(doc_path, domain + '.json')) as f:
                for line in f:
                    field = json.loads(line)
                    page_id = field['document_id']
                    doc[page_id] = field
        return doc

    train_doc = load_entities(data_dir, train_domain)
    heldout_train_doc = load_entities(data_dir, heldout_train_domain)
    heldout_train_unseen_doc = load_entities(data_dir,
                                             heldout_train_unseen_domain)
    val_doc = load_entities(data_dir, val_domain)
    test_doc = load_entities(data_dir, test_domain)

    return samples_train, samples_heldout_train_seen, \
           samples_heldout_train_unseen, samples_val, samples_test, \
           train_doc, heldout_train_doc, heldout_train_unseen_doc, \
           heldout_train_unseen_doc, val_doc, test_doc


class Data:
    def __init__(self, train_doc, val_doc, test_doc, tokenizer, max_len,
                 train_mention, val_mention, test_mention):
        self.train_doc = train_doc
        self.val_doc = val_doc
        self.test_doc = test_doc
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train_men = train_mention
        self.val_men = val_mention
        self.test_men = test_mention

        self.batch_size = 512
        self.num_workers = 16

    def get_train_split_loaders(self):
        train_en_set = EntitySet(self.tokenizer, self.train_doc, self.max_len)
        train_men_set = MentionSet(self.tokenizer, self.train_men, self.train_doc, self.max_len)

        train_en_loader = DataLoader(train_en_set, self.batch_size, shuffle=False, num_workers=self.num_workers)
        train_men_loader = DataLoader(train_men_set, self.batch_size, shuffle=False, num_workers=self.num_workers)
        return train_en_loader, train_men_loader

    def get_valid_split_loaders(self):
        val_en_set = EntitySet(self.tokenizer, self.val_doc, self.max_len)
        val_men_set = MentionSet(self.tokenizer, self.val_men, self.val_doc, self.max_len)

        val_en_loader = DataLoader(val_en_set, self.batch_size, shuffle=False, num_workers=self.num_workers)
        val_men_loader = DataLoader(val_men_set, self.batch_size, shuffle=False, num_workers=self.num_workers)
        return val_en_loader, val_men_loader

    def get_test_split_loaders(self):
        test_en_set = EntitySet(self.tokenizer, self.test_doc, self.max_len)
        test_men_set = MentionSet(self.tokenizer, self.test_men, self.test_doc, self.max_len)

        test_en_loader = DataLoader(test_en_set, self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_men_loader = DataLoader(test_men_set, self.batch_size, shuffle=False, num_workers=self.num_workers)
        return test_en_loader, test_men_loader


class ZeshelDataset(Dataset):
    def __init__(self, tokenizer, mentions, doc, max_len, candidates, num_rands, type_cands):
        self.mentions = MentionSet(tokenizer, mentions, doc, max_len)
        self.entities = EntitySet(tokenizer, doc, max_len)
        self.candidates = candidates
        self.num_rands = num_rands
        self.type_cands = type_cands

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, index):
        """

        :param index: The index of mention
        :return: mention_token_ids,mention_masks,entity_token_ids,entity_masks : 1 X L
                entity_hard_token_ids, entity_hard_masks: k X L  (k<=10)
        """
        mention_token_ids, mention_masks, _ = self.mentions[index]

        
        # process entity
        target_page_id = self.mentions.processed[index]['label_document_id']
        all_entities = list(self.entities.ent_id_to_idx.keys())

        entity_token_ids, entity_masks = self.entities[self.entities.ent_id_to_idx[target_page_id]]
        candidate_token_ids = [entity_token_ids]
        candidate_masks = [entity_masks]


        if self.type_cands == 'hard_and_random_negative':
            random_cands_pool = set(all_entities) - set([target_page_id])
            rand_cands = random.sample(random_cands_pool, self.num_rands)
            for page_id in rand_cands:
                rand_entity_token_ids, rand_entity_masks = self.entities[self.entities.ent_id_to_idx[page_id]]
                candidate_token_ids.append(rand_entity_token_ids)
                candidate_masks.append(rand_entity_masks)

            # process hard negatives
            hard_negs = self.candidates[index]
            for idx in hard_negs:
                hard_entity_token_ids, hard_entity_masks = self.entities[idx]
                candidate_token_ids.append(hard_entity_token_ids)
                candidate_masks.append(hard_entity_masks)


        elif self.type_cands == 'distributed_negative':
            distributed_cands = self.candidates[index]
            for idx in distributed_cands:
                candidate_entity_token_ids, candidate_entity_masks = self.entities[idx]
                candidate_token_ids.append(candidate_entity_token_ids)
                candidate_masks.append(candidate_entity_masks)
        else:
            raise ValueError('wrong type candidates')

        
        candidate_token_ids = torch.stack(candidate_token_ids, dim=0).long()
        candidate_masks = torch.stack(candidate_masks, dim=0).long()
        return mention_token_ids, mention_masks, candidate_token_ids, \
            candidate_masks