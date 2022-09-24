import torch
from model import DualEncoder
# from fast_data import *
import logging
import argparse
import numpy as np
import os
import random
import torch.nn as nn
from torch.nn.parallel import data_parallel
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, AdamW, \
    get_linear_schedule_with_warmup, get_constant_schedule
# import faiss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datetime import datetime
# import faiss

# from box_embeddings.parameterizations.delta_box_tensor import MinDeltaBoxTensor
from box_embeddings.parameterizations.sigmoid_box_tensor import SigmoidBoxTensor
from box_embeddings.modules.volume import Volume
from box_embeddings.modules.intersection import Intersection

from data.base import Data, MentionSet, EntitySet, load_data
from utils import get_all_entity_embeds


def load_domain_data(data_dir):
    """

    :param data_dir: train_data_dir if args.train else eval_data_dir
    :return: mentions, entities,doc
    """
    print('begin loading data')
    men_path = os.path.join(data_dir, 'mentions')

    def get_domains(part):
        domains = set()
        with open(os.path.join(men_path, '%s.json' % part)) as f:
            for line in f:
                field = json.loads(line)
                domains.add(field['corpus'])
        return list(domains)

    def load_domain_mentions(domain, part):
        domain_mentions = []
        with open(os.path.join(men_path, '%s.json' % part)) as f:
            for line in f:
                field = json.loads(line)
                if field['corpus'] == domain:
                    domain_mentions.append(field)
        return domain_mentions

    def load_domain_entities(data_dir, domain):
        """

        :param domains: list of domains
        :return: all the entities in the domains
        """
        doc = {}
        doc_path = os.path.join(data_dir, 'documents')
        with open(os.path.join(doc_path, domain + '.json')) as f:
            for line in f:
                field = json.loads(line)
                page_id = field['document_id']
                doc[page_id] = field
        return doc

    val_domains = get_domains('val')
    test_domains = get_domains('test')
    val_mentions = []
    test_mentions = []
    val_doc = []
    test_doc = []
    for val_domain in val_domains:
        domain_mentions = load_domain_mentions(val_domain, 'val')
        domain_entities = load_domain_entities(data_dir, val_domain)
        val_mentions.append(domain_mentions)
        val_doc.append(domain_entities)
    for test_domain in test_domains:
        domain_mentions = load_domain_mentions(test_domain, 'test')
        domain_entities = load_domain_entities(data_dir, test_domain)
        test_mentions.append(domain_mentions)
        test_doc.append(domain_entities)

    return val_mentions, test_mentions, val_doc, test_doc

def chunked_process(men_boxes, all_en_boxes, box_int, box_vol, k, device):
    n_mens = men_boxes.box_shape[0]
    pred_labels = torch.zeros((n_mens, k), dtype=int, device=device)

    for i in range(n_mens):
        men_box = men_boxes[[i]]
        int_log_prob = box_vol(box_int(men_box, all_en_boxes)) - box_vol(men_box)
        pred_labels[i, :] = torch.topk(torch.exp(int_log_prob), k=k)[1]

    return pred_labels


def macro_evaluate(mention_loaders, model, en_embeds, k, device):
    return None
    # model.eval()
    # macro_recall = 0
    # macro_acc = 0
    # macro_normalized_acc = 1
    # if hasattr(model, 'module'):
    #     encoder = model.module.mention_encoder
    #     box_volume = model.module.box_volume
    #     box_intersection = model.module.box_intersection
    # else:
    #     encoder = model.mention_encoder
    #     box_volume = model.box_volume
    #     box_intersection = model.box_intersection

    # for i in range(len(mention_loaders)):
    #     mention_loader = mention_loaders[i]
    #     entity_embed = en_embeds[i]
    #     r_k = 0
    #     acc = 0
    #     nb_samples = 0
    #     normalized_acc = 0
    #     with torch.no_grad():
    #         for j, batch in enumerate(mention_loader):
    #             men_embeds = encoder(input_ids=batch[0].to(device),
    #                                  attention_mask=batch[1].to(device))[0][:, 0, :]

    #             men_boxes = SigmoidBoxTensor.from_vector(men_embeds.to(device))
    #             en_boxes = SigmoidBoxTensor.from_vector(entity_embed.to(device))

    #             # Perform one by one to prevent OOM issues
    #             top_k = chunked_process(men_boxes, en_boxes, box_intersection, box_volume, k, device)

    #             # logits = men_embeds @ entity_embed.t().to(device)
    #             labels = batch[2].to(device)
    #             preds = top_k[:, 0]
    #             r_k += (top_k == labels.to(device)).sum().item()
    #             nb_samples += men_embeds.size(0)
    #             acc += (preds == labels.squeeze(1).to(device)).sum().item()
    #     r_k /= nb_samples
    #     acc /= nb_samples
    #     if r_k!=0:
    #         normalized_acc = acc / r_k
    #     macro_recall += r_k
    #     macro_acc += acc
    #     macro_normalized_acc += normalized_acc
    # macro_recall /= len(mention_loaders)
    # macro_acc /= len(mention_loaders)
    # macro_normalized_acc /= len(mention_loaders)

    # return macro_recall, macro_acc, macro_normalized_acc


def micro_evaluate(men_loader, model, all_en_embeds, k, device):
    model.eval()
    nb_samples = 0
    r_k = 0
    acc = 0
    if hasattr(model, 'module'):
        encoder = model.module.mention_encoder
        box_volume = model.module.box_volume
        box_intersection = model.module.box_intersection
    else:
        encoder = model.mention_encoder
        box_volume = model.box_volume
        box_intersection = model.box_intersection

    all_en_boxes =  SigmoidBoxTensor.from_vector(all_en_embeds.to(device).flatten(1))
    with torch.no_grad():
        for i, batch in enumerate(men_loader):
            men_embeds = encoder(input_ids=batch[0].to(device),
                                 attention_mask=batch[1].to(device))[0][:, :2, :]
            
            men_boxes = SigmoidBoxTensor.from_vector(men_embeds.to(device).flatten(1))

            # Perform one by one to prevent OOM issues
            top_k = chunked_process(men_boxes, all_en_boxes, box_intersection, box_volume, k, device)

            labels = batch[2].to(device)
            preds = top_k[:, 0]
            r_k += (top_k == labels.to(device)).sum().item()
            nb_samples += men_embeds.size(0)
            acc += (preds == labels.squeeze(1).to(device)).sum().item()

    r_k /= nb_samples
    acc /= nb_samples

    return r_k, acc


def main(args):
    data_dir = args.data_dir
    LOG_FORMAT = "%(levelname)s:  %(message)s"
    logging.basicConfig(

        level=logging.DEBUG,
        format=LOG_FORMAT)
    logger = logging.getLogger()
    # load data and initialize model and dataset
    if args.eval_method == 'macro':
        val_mentions, test_mentions, val_doc, test_doc = load_domain_data(
            data_dir)

        # get model and tokenizer
        print('len of test mentions')
        print(len(test_mentions))
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        encoder = BertModel.from_pretrained('bert-base-uncased')
        # encoder=MLPEncoder(args.max_len)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DualEncoder(encoder, device, args.use_local_nce).to(
            device)

        dp = torch.cuda.device_count() > 1
        model_path = args.model
        package = torch.load(
            model_path) if device.type == 'cuda' else torch.load(
            model_path,
            map_location=torch.device('cpu'))
        if dp:
            from collections import OrderedDict

            state_dict = package['sd']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
        else:
            new_state_dict = package['sd']
        model.load_state_dict(new_state_dict)
        model.eval()
        max_len = args.max_len
        B = args.eval_batchsize
        k = args.k
        val_mention_loaders = []
        val_en_embeds = []
        for i in range(len(val_mentions)):
            val_domain_mentions = val_mentions[i]
            val_entities = val_doc[i]
            val_domain_mention_set = MentionSet(tokenizer, val_domain_mentions,
                                                val_entities, max_len)
            val_domain_entity_set = EntitySet(tokenizer, val_entities, max_len)
            val_mention_loaders.append(DataLoader(val_domain_mention_set, B,
                                                  shuffle=False))
            val_entity_loader = DataLoader(val_domain_entity_set, B,
                                           shuffle=False)
            val_en_embeds.append(
                get_all_entity_embeds(val_entity_loader, model, device, dp))
        r_k, acc, n_acc = macro_evaluate(val_mention_loaders, model,
                                         val_en_embeds,
                                         k,
                                         device)
        logger.info(' val recall@{:d} : {:8.4f}'.format(k, r_k))
        logger.info(' val acc {:8.4f}'.format(acc))
        logger.info(' val normalized acc {:8.4f}'.format(n_acc))

        test_mention_loaders = []
        test_en_embeds = []
        for i in range(len(test_mentions)):
            test_domain_mentions = test_mentions[i]
            test_entities = test_doc[i]
            test_domain_mention_set = MentionSet(tokenizer,
                                                 test_domain_mentions,
                                                 test_entities, max_len)
            test_domain_entity_set = EntitySet(tokenizer, test_entities,
                                               max_len)
            test_mention_loaders.append(DataLoader(test_domain_mention_set, B,
                                                   shuffle=False))
            test_entity_loader = DataLoader(test_domain_entity_set, B,
                                            shuffle=False)
            test_en_embeds.append(
                get_all_entity_embeds(test_entity_loader, model, device, dp))
        r_k, acc, n_acc = macro_evaluate(test_mention_loaders, model,
                                         test_en_embeds,
                                         k,
                                         device)
        logger.info(' test recall@{:d} : {:8.4f}'.format(k, r_k))
        logger.info(' test acc {:8.4f}'.format(acc))
        logger.info(' test normalized acc {:8.4f}'.format(n_acc))
    elif args.eval_method == 'micro':
        samples_train, samples_heldout_train_seen, \
        samples_heldout_train_unseen, samples_val, samples_test, \
        train_doc, heldout_train_doc, heldout_train_unseen_doc, \
        heldout_train_unseen_doc, val_doc, test_doc = load_data(args.data_dir)

        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # encoder = BertModel.from_pretrained('bert-base-uncased')

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        encoder = AutoModel.from_pretrained('bert-base-uncased')


        data = Data(train_doc, val_doc, test_doc, tokenizer, args.max_len,
                    samples_train, samples_val, samples_test)

        val_en_loader, val_men_loader = data.get_valid_split_loaders()
        test_en_loader, test_men_loader = data.get_test_split_loaders()

        # train_en_loader, val_en_loader, test_en_loader, train_men_loader, \
        # val_men_loader, test_men_loader = data.get_loaders(args.B)

        # get model and tokenizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DualEncoder(encoder, args.use_local_nce, args.vol_temp, args.vol_int_temp, args.int_temp).to(
            device)

        dp = torch.cuda.device_count() > 1
        model_path = args.model
        package = torch.load(
            model_path) if device.type == 'cuda' else torch.load(
            model_path,
            map_location=torch.device('cpu'))
        if dp:
            from collections import OrderedDict

            state_dict = package['sd']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
        else:
            new_state_dict = package['sd']
        model.load_state_dict(new_state_dict)
        model.eval()
        all_val_en_embeds = get_all_entity_embeds(val_en_loader, model, device, dp)
        val_result = micro_evaluate(val_men_loader, model, all_val_en_embeds, args.k, device)
        logger.info(' val recall@{:d} : {:8.4f}'
                    '| val accuracy : {:8.4f}'.format(args.k, val_result[0],
                                                      val_result[1]))
        all_test_en_embeds = get_all_entity_embeds(test_en_loader, model,
                                                   device, dp)
        test_result = micro_evaluate(test_men_loader, model, all_test_en_embeds,
                                     args.k,
                                     device)
        logger.info(' test recall@{:d} : {:8.4f}'
                    '| test accuracy : {:8.4f}'.format(args.k, test_result[0],
                                                       test_result[1]))
    else:
        raise ValueError('wrong evaluate method')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='model path')
    parser.add_argument('--max_len', type=int, default=128,
                        help='max length of the mention input '
                             'and the entity input')
    parser.add_argument('--data_dir', type=str,
                        help='the  data directory')

    parser.add_argument('--B', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--k', type=int, default=64,
                        help='recall@k when evaluate')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [%(default)d]')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers [%(default)d]')
    parser.add_argument('--gpus', default='', type=str,
                        help='GPUs separated by comma [%(default)s]')
    parser.add_argument('--eval_method', default='micro', type=str,
                        choices=['macro', 'micro'],
                        help='the evaluate method')
    parser.add_argument('--eval_batchsize', default=512, type=int,
                        help='the batch size when evaluate')
    parser.add_argument('--use_local_nce', action='store_true',
                        help='use local NCE loss? ')
    parser.add_argument('--vol_temp', type=float, default=1,
                        help='Volume temperature parameter for boxes [%(default)g]')
    parser.add_argument('--vol_int_temp', type=float, default=1e-5,
                        help='Volume intersection temperature parameter for boxes [%(default)g]')
    parser.add_argument('--int_temp', type=float, default=1e-5,
                        help='Intersection temperature parameter for boxes [%(default)g]')

    args = parser.parse_args()
    # Set environment variables before all else.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus  # Sets torch.cuda behavior
    main(args)
