from email.policy import default
import os
from collections import defaultdict
import torch
from torch.utils.data import Dataset

import datasets as hds
from datasets.fingerprint import Hasher

def get_entity_window(entry, tokenizer, max_len, ENT):
    # page_id = en_page_id
    en_title = entry['title']
    en_text = entry['text']
    max_len = max_len - 2  # cls , sep
    en_title_tokens = tokenizer.tokenize(en_title)
    en_text_tokens = tokenizer.tokenize(en_text)
    window = (en_title_tokens + [ENT] + en_text_tokens)[:max_len]
    return window


def process_entity_entry(entity, tokenizer, max_len, ENT):
    # process entity
    entity_window = get_entity_window(entity, tokenizer, max_len, ENT)
    entity_dict = tokenizer.encode_plus(
        entity_window, add_special_tokens=True,
        max_length=max_len, padding='max_length',
        truncation=True, is_split_into_words=True
    )

    return {
        'entity_token_ids': entity_dict['input_ids'],
        'entity_masks': entity_dict['attention_mask']
    }


def bulk_process_entities(collected_docs, tokenizer, max_len, ENT, ent_id_to_idx):

    collected_docs = collected_docs.map(
        process_entity_entry,
        num_proc=16,
        fn_kwargs={
            'tokenizer': tokenizer,
            'max_len': max_len,
            'ENT': ENT
        },
        remove_columns=['title', 'text']
    )
    return collected_docs

def func(e, ent_id_to_idx):
    e['idx'] = ent_id_to_idx[e['document_id']]
    return e

class EntitySet(Dataset):
    def __init__(self, tokenizer, doc, max_len):
        self.tokenizer = tokenizer
        self.doc = doc
        self.max_len = max_len  # the max  length of input (mention or entity)
        self.ENT = '[unused2]'
        self.all_entities = list(self.doc.keys())
        self.ent_id_to_idx = {ent_id: idx for idx,
                              ent_id in enumerate(doc.keys())}

        self.cache_dir = '/dev/shm/zeshel_cache'
        fingerprint = Hasher.hash(
            list(sorted(doc.keys()))
        )

        print('searching', fingerprint)
        if not os.path.exists(f'{self.cache_dir}/{fingerprint}'):
            print('valid cache not found; loading...')
            processed_docs = defaultdict(list)
            for entry in self.doc.values():
                for k, v in entry.items():
                    processed_docs[k].append(v)

            processed_docs = hds.Dataset.from_dict(processed_docs)
            processed_docs.save_to_disk(f'{self.cache_dir}/{fingerprint}')


        processed_docs = hds.Dataset.load_from_disk(f'{self.cache_dir}/{fingerprint}')
        processed_docs = bulk_process_entities(
            processed_docs, self.tokenizer, self.max_len, self.ENT, self.ent_id_to_idx
        )
          
        self.processed = processed_docs.map(
            func, fn_kwargs={'ent_id_to_idx': self.ent_id_to_idx}
        ).sort('idx')

    def __len__(self):
        return len(self.doc)

    def __getitem__(self, index):
        """

        :param index: The index of mention
        :return: mention_token_ids,mention_masks,entity_token_ids,entity_masks : 1 X L
                entity_hard_token_ids, entity_hard_masks: k X L  (k<=10)
        """
        return (
            torch.tensor(self.processed[index]['entity_token_ids']).long(),
            torch.tensor(self.processed[index]['entity_masks']).long()
        )