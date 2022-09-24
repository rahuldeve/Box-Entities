import os
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader

import datasets as hds
from datasets.fingerprint import Hasher

def help_mention_window(ctx_l, mention, ctx_r, max_len, Ms, Me):
    if len(mention) >= max_len:
        window = mention[:max_len]
        return window
    leftover = max_len - len(mention) - 2  # [Ms] token and [Me] token
    leftover_hf = leftover // 2
    if len(ctx_l) > leftover_hf:
        ctx_l_len = leftover_hf if len(
            ctx_r) > leftover_hf else leftover_hf - len(ctx_r)
    else:
        ctx_l_len = len(ctx_l)
    window = ctx_l[-ctx_l_len:] + [Ms] + mention + [Me] + ctx_r
    window = window[:max_len]
    return window


def get_mention_window(mention, tokenizer, doc, max_len, Ms, Me):
    page_id = mention['context_document_id']
    tokens = doc[page_id]['text'].split()
    max_mention_len = max_len - 2  # cls and sep

    # assert men == context_tokens[start_index:end_index]
    ctx_l = tokens[max(0, mention['start_index'] - max_mention_len - 1):
                   mention['start_index']]
    ctx_r = tokens[mention['end_index'] + 1:
                   mention['end_index'] + max_mention_len + 2]
    men = tokens[mention['start_index']:mention['end_index'] + 1]

    ctx_l = ' '.join(ctx_l)
    ctx_r = ' '.join(ctx_r)
    men = ' '.join(men)
    men = tokenizer.tokenize(men)
    ctx_l = tokenizer.tokenize(ctx_l)
    ctx_r = tokenizer.tokenize(ctx_r)
    return help_mention_window(ctx_l, men, ctx_r, max_mention_len, Ms, Me)


def process_mention_entry(mention, tokenizer, doc, max_len, Ms, Me):
    doc = dict(doc)
    mention_window = get_mention_window(mention, tokenizer, doc, max_len, Ms, Me)
    mention_encoded_dict = tokenizer.encode_plus(
        mention_window, add_special_tokens=True,
        max_length=max_len, padding='max_length',
        truncation=True, is_split_into_words=True
    )
    mention_token_ids = mention_encoded_dict['input_ids']
    mention_masks = mention_encoded_dict['attention_mask']
    
    return {
        'mention_token_ids': mention_token_ids,
        'mention_masks': mention_masks,
        # 'label': label
    }


def bulk_process_mentions(collected_mentions, tokenizer, doc, max_len, Ms, Me):
    # doc = dict(doc)
    collected_mentions = collected_mentions.map(
        process_mention_entry,
        num_proc=8,
        fn_kwargs={
            'tokenizer': tokenizer,
            'doc': doc,
            'max_len': max_len,
            'Ms': Ms,
            'Me': Me
        },
        remove_columns=[
            'category', 'text', 'end_index', 'context_document_id',
            'corpus', 'start_index'
        ]
    )

    return collected_mentions


class MentionSet(Dataset):
    def __init__(self, tokenizer, mentions, doc, max_len):
        self.tokenizer = tokenizer
        self.mentions = mentions
        self.doc = doc
        self.max_len = max_len  # the max  length of input (mention or entity)
        self.Ms = '[unused0]'
        self.Me = '[unused1]'

        self.cache_dir = '/dev/shm/zeshel_cache'
        fingerprint = Hasher.hash([
            mentions,
            list(sorted(doc.keys()))
        ])

        print('searching', fingerprint)
        if not os.path.exists(f'{self.cache_dir}/{fingerprint}'):
            print('valid cache not found')
            processed_mentions = defaultdict(list)
            for entry in mentions:
                for k, v in entry.items():
                    processed_mentions[k].append(v)

            processed_mentions = hds.Dataset.from_dict(processed_mentions)
            processed_mentions.save_to_disk(f'{self.cache_dir}/{fingerprint}')

            

        processed_mentions = hds.Dataset.load_from_disk(
            f'{self.cache_dir}/{fingerprint}'
        )

        processed_mentions = bulk_process_mentions(
            processed_mentions, 
            self.tokenizer, 
            list(sorted(self.doc.items())), 
            # self.doc,
            self.max_len, 
            self.Ms, self.Me
        )

        men_idx_order = {
            men_idx: idx 
            for idx,men_idx in enumerate(map(lambda e: e['mention_id'], mentions))
        }

        entity_order = {
            ent_idx: idx for idx, ent_idx in enumerate(doc.keys())
        }

        def add_label_column(mention_entry):
            ent_idx = mention_entry['label_document_id']
            mention_entry['label'] = entity_order[ent_idx]
            return mention_entry

        def add_idx_col(e):
            e['idx'] = men_idx_order[e['mention_id']]
            return e


        processed_mentions = processed_mentions.map(
            add_label_column
        ).map(
            add_idx_col
        ).sort('idx')


        self.processed = processed_mentions
        

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, index):
        """

        :param index: The index of mention
        :return: mention_token_ids,mention_masks,entity_token_ids,entity_masks : 1 X L
                entity_hard_token_ids, entity_hard_masks: k X L  (k<=10)
        """
        return (
            torch.tensor(self.processed[index]['mention_token_ids']).long(), 
            torch.tensor(self.processed[index]['mention_masks']).long(),
            torch.tensor([self.processed[index]['label']]).long()
        )
