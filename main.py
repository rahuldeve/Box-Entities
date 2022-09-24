import argparse
from loguru import logger

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule


from utils import set_seeds, count_parameters
from utils import get_all_entity_embeds, get_hard_negative
from utils import evaluate
from data.base import load_data, Data, ZeshelDataset
from model import DualEncoder

from tqdm import tqdm

from datasets.utils.logging import set_verbosity_error
set_verbosity_error()

from pprint import pformat



def configure_optimizer(args, model, num_train_examples):
    # https://github.com/google-research/bert/blob/master/optimization.py#L25
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)

    num_train_steps = int(
        num_train_examples / args.B / args.gradient_accumulation_steps * args.epochs
    )
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps)

    return optimizer, scheduler, num_train_steps, num_warmup_steps


def configure_optimizer_simple(args, model, num_train_examples):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_train_steps = int(
        num_train_examples / args.B / args.gradient_accumulation_steps * args.epochs
    )
    num_warmup_steps = 0
    scheduler = get_constant_schedule(optimizer)
    return optimizer, scheduler, num_train_steps, num_warmup_steps



def prep_train_dataloader(
    train_en_loader, train_men_loader, samples_train, train_doc, model, tokenizer,
    num_hards, num_rands, device, dp 
):
    all_train_en_embeds = get_all_entity_embeds(train_en_loader, model, device, dp)
    if args.type_cands == 'distributed_negative':
        # if args.use_smoothing:
        return None
        # candidates = get_distributed_candidates(
        #     train_men_loader, model, all_train_en_embeds,
        #     args.num_cands, device, dp, args.use_smoothing
        # )
        
    elif args.type_cands == 'hard_and_random_negative':
        candidates = get_hard_negative(
            train_men_loader, model, all_train_en_embeds, num_hards,
            device, dp
        )
    else:
        raise ValueError('type candidates wrong')

    
    train_set = ZeshelDataset(
        tokenizer, samples_train, train_doc,
        args.max_len, candidates, num_rands, args.type_cands
    )

    train_loader = DataLoader(
        train_set, args.B, shuffle=True, drop_last=False, num_workers=32
    )

    return train_loader


def train_one_epoch(model, train_loader, optimizer, scheduler, epoch_num):

    total_train_loss = 0.0
    for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

        model.train()
        loss = model(*batch)[0]

        if len(args.gpus) > 1:
            loss = loss.mean()

        loss.backward()
        total_train_loss += loss.item()


        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            if (step + 1) % args.logging_steps == 0:
                curr_avg_loss = total_train_loss / (step + 1)
                logger.info(
                    f'Epoch {epoch_num} | Batch {step} | Current average Loss {curr_avg_loss}'
                )

    return total_train_loss / len(train_loader)





def main(args):
    # Initial configurations
    set_seeds(args.seed)
    logger.add(args.logfile_path)

    logger.info(pformat(vars(args)))


    # Load initial datasets and dataloaders
    (
        samples_train, samples_heldout_train_seen,
        samples_heldout_train_unseen, samples_val, samples_test,
        train_doc, heldout_train_doc, heldout_train_unseen_doc,
        heldout_train_unseen_doc, val_doc, test_doc
    ) = load_data(args.data_dir)

    num_rands = int(args.num_cands * args.cands_ratio)
    num_hards = args.num_cands - num_rands

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    data = Data(
        train_doc, val_doc, test_doc, tokenizer, args.max_len, 
        samples_train, samples_val, samples_test
    )

    train_en_loader, train_men_loader = data.get_train_split_loaders()
    val_en_loader, val_men_loader = data.get_valid_split_loaders()
    # test_en_loader, test_men_loader = data.get_test_split_loaders()


    # Create and prepare model for training
    encoder = AutoModel.from_pretrained(args.pretrained_model)
    model = DualEncoder(
        encoder,
        args.vol_temp, args.vol_int_temp, args.int_temp
    )
    model = nn.DataParallel(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # configure optimizer
    num_train_samples = len(samples_train)
    if args.simpleoptim:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer_simple(args, model, num_train_samples)
    else:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer(args, model, num_train_samples)


    logger.info('***** train *****')
    logger.info('# train samples: {:d}'.format(num_train_samples))
    logger.info('# epochs: {:d}'.format(args.epochs))
    logger.info(' batch size: {:d}'.format(args.B))
    logger.info(' gradient accumulation steps {:d}'
                ''.format(args.gradient_accumulation_steps))
    logger.info(
        ' effective training batch size with accumulation: {:d}'
        ''.format(args.B * args.gradient_accumulation_steps))
    logger.info(' # training steps: {:d}'.format(num_train_steps))
    logger.info(' # warmup steps: {:d}'.format(num_warmup_steps))
    logger.info(' learning rate: {:g}'.format(args.lr))
    logger.info(' # parameters: {:d}'.format(count_parameters(model)))



    # step_num = 0
    # tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    best_val_perf = float('-inf')
    for epoch in range(1, args.epochs + 1):
        logger.info('\nEpoch {:d}'.format(epoch))

        with torch.no_grad():
            train_loader = prep_train_dataloader(
                train_en_loader, train_men_loader, samples_train, train_doc,
                model, tokenizer, num_hards, num_rands, device, dp=True
            )
            # torch.cuda.empty_cache()

        avg_train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch)
        with torch.no_grad():
            model.zero_grad()
            model.eval()
            eval_result = evaluate(val_en_loader, val_men_loader, model, args.k, device)

        logger.info(
            f'Done with epoch {epoch} | average train loss {avg_train_loss} | \
            validation recall {eval_result[0]:8.4f} | validation accuracy {eval_result[1]:8.4f}'
        )


        if args.eval_criterion == 'recall':
            if eval_result[0] >= best_val_perf:
                logger.info('------- new best val perf: {:g} --> {:g} '
                            ''.format(best_val_perf, eval_result[0]))
                best_val_perf = eval_result[0]
                torch.save({'opt': args, 'sd': model.state_dict(),
                            'perf': best_val_perf}, args.model)
            else:
                logger.info('')
        else:
            if eval_result[1] >= best_val_perf:
                logger.info('------- new best val perf: {:g} --> {:g} '
                            ''.format(best_val_perf, eval_result[1]))
                best_val_perf = eval_result[1]
                torch.save({'opt': args, 'sd': model.state_dict(),
                            'perf': best_val_perf}, args.model)
            else:
                logger.info('')
        

    # # test model on test dataset
    # package = torch.load(args.model) if device.type == 'cuda' \
    #     else torch.load(args.model, map_location=torch.device('cpu'))
    # if dp:
    #     from collections import OrderedDict
    #     state_dict = package['sd']
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         name = k[7:]
    #         new_state_dict[name] = v
    # else:
    #     new_state_dict = package['sd']
    # # encoder=MLPEncoder(args.max_len)
    # if args.pre_model == 'Bert':
    #     # encoder = BertModel.from_pretrained('bert-base-uncased')
    #     encoder = AutoModel.from_pretrained('microsoft/MiniLM-L12-H384-uncased')
    # if args.pre_model == 'Roberta':
    #     encoder = RobertaModel.from_pretrained('roberta-base')

    # model = DualEncoder(encoder, device,args.use_local_nce).to(device)
    # model.load_state_dict(new_state_dict)
    # model.eval()
    # all_test_en_embeds = get_all_entity_embeds(test_en_loader, model,
    #                                            device, dp)
    # test_result = evaluate(test_men_loader, model, all_test_en_embeds, args.k,
    #                        device)
    # logger.info(' test recall@{:d} : {:8.4f}'
    #             '| test accuracy : {:8.4f}'.format(args.k, test_result[0],
    #                                                test_result[1]))

    






if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str, 
        help='save path for trained model'
    )

    parser.add_argument('--data_dir', type=str, help='data directory location')

    parser.add_argument('--logfile_path', type=str, help='path of the generated logfile')

    parser.add_argument(
        '--max_len', type=int, default=128, 
        help='max length of the mention input and the entity input'
    )

    # parser.add_argument(
    #     '--num_hards', type=int, default=10, 
    #     help='the number of the nearest neighbors we use to construct hard negatives'
    # )

    parser.add_argument(
        '--type_cands', type=str, default='hard_and_random_negative',
        choices=['hard_and_random_negative', 'distributed_negative'],
        help='the type of negative we use during training'
    )

    

    parser.add_argument('--B', type=int, default=128, help='batch size')

    parser.add_argument(
        '--lr', type=float, default=2e-5,
        choices=[5e-6, 1e-5, 2e-5, 5e-5, 0.0002, 0.002],
        help='the learning rate'
    )


    parser.add_argument('--epochs', type=int, default=3, help='the number of training epochs')

    parser.add_argument('--k', type=int, default=64, help='recall@k when evaluate')


    parser.add_argument(
        '--warmup_proportion', type=float, default=0.1,
        help='proportion of training steps to perform linear learning rate warmup for [%(default)g]'
    )


    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay [%(default)g]')

    parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='epsilon for Adam optimizer [%(default)g]')

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='num gradient accumulation steps [%(default)d]')

    parser.add_argument('--seed', type=int, default=42, help='random seed [%(default)d]')

    # parser.add_argument('--num_workers', type=int, default=0, help='num workers [%(default)d]')

    parser.add_argument('--simpleoptim', action='store_true', help='simple optimizer (constant schedule, no weight decay?')

    parser.add_argument('--use_local_nce', action='store_true', help='use local NCE loss? ')

    parser.add_argument('--clip', type=float, default=1, help='gradient clipping [%(default)g]')

    parser.add_argument('--logging_steps', type=int, default=1000, help='num logging steps [%(default)d]')

    parser.add_argument('--gpus', default='', type=str, help='GPUs separated by comma [%(default)s]')

    parser.add_argument(
        '--eval_criterion', type=str, default='recall', 
        choices=['recall', 'accuracy'],
        help='the criterion for selecting model'
    )


    parser.add_argument('--pretrained_model', type=str, help='huggingface identifier of the pretrained model')
    # parser.add_argument('--pre_model', default='Bert',
    #                     choices=['Bert', 'Roberta'],
    #                     type=str, help='the encoder for train'
    # )


    # parser.add_argument('--lambd', default=0.9, type=float, help='the ratio of random loss')

    parser.add_argument(
        '--cands_ratio', default=1.0, type=float,
        help='the ratio between random candidates and hard candidates'
    )

    parser.add_argument(
        '--num_cands', default=16, type=int, 
        help='the total number of candidates'
    )


    parser.add_argument(
        '--use_smoothing', action='store_true',
        help='use 0.75 smoothing when sampling negatives according to model distribution'
    )

    parser.add_argument('--vol_temp', type=float, default=1.0, help='Volume temperature parameter for boxes [%(default)g]')
    parser.add_argument('--vol_int_temp', type=float, default=1e-5, help='Volume intersection temperature parameter for boxes [%(default)g]')
    parser.add_argument('--int_temp', type=float, default=1e-5, help='Intersection temperature parameter for boxes [%(default)g]')

    args = parser.parse_args()
    main(args)