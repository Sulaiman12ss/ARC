import argparse
import json
import logging
import os

import torch
from transformers import TrainingArguments, set_seed  # type: ignore

import wandb
from data.arc import prepare_arc_tasks
from data.dataset import ARCDataset, HuggingFaceDataset, get_tokenized_datasets
from models.model import get_raw_gpt2
from models.tokenizer import TaskTokenizer
from train import train
from utils import count_trainable_parameters


def parse_args():
    parser = argparse.ArgumentParser(
        description='load gpt model from huggingface and train/finetune on custom dataset',
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--output_dir', metavar='PATH', type=str, required=True)
    parser.add_argument('--train_dataset', metavar='DATASET', type=str, required=True)
    parser.add_argument('--num_steps_per_epoch', metavar='DATASET', type=int, default=None)
    parser.add_argument('--run_name', metavar='W&B', type=str, default='run')

    parser.add_argument('--model_type', metavar='MODEL', type=str, default='gpt2')
    parser.add_argument('--pretrained_path_hf', metavar='MODEL', type=str, default=None)
    parser.add_argument('--from_checkpoint', metavar='MODEL', type=str, default=None)

    parser.add_argument('--add_row_sep', action='store_true')
    parser.add_argument('--segregate', metavar='COLLATE', type=str, default='row')
    parser.add_argument('--target_only_test_output', action='store_true')

    parser.add_argument('--train_batch_size', metavar='LOADER', type=int, default=1)
    parser.add_argument('--test_batch_size', metavar='LOADER', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', metavar='LOADER', type=int, default=1)

    parser.add_argument('--hidden_size', metavar='MODEL', type=int, default=256)
    parser.add_argument('--num_hidden_layers', metavar='MODEL', type=int, default=8)
    parser.add_argument('--num_attention_heads', metavar='MODEL', type=int, default=8)
    parser.add_argument('--max_position_embeddings', metavar='MODEL', type=int, default=24000)

    parser.add_argument('--learning_rate', metavar='TRAIN', type=float, default=3e-4)
    parser.add_argument('--weight_decay', metavar='TRAIN', type=float, default=0.01)
    parser.add_argument('--num_steps', metavar='TRAIN', type=int, default=40000)
    parser.add_argument(
        '--scheduler_type', metavar='TRAIN', type=str, default='constant', choices=['constant', 'linear']
    )

    parser.add_argument('--log_every', metavar='TRAIN', type=int, default=50)
    parser.add_argument('--eval_every', metavar='TRAIN', type=int, default=500)
    parser.add_argument('--save_every', metavar='TRAIN', type=int, default=5000)

    parser.add_argument(
        '--memory_tokens_strategy', metavar='TRAIN_EXTRA', type=str, default='left', choices=['left', 'right']
    )
    parser.add_argument('--num_memory_tokens', metavar='TRAIN_EXTRA', type=int, default=0)

    parser.add_argument('--use_bf16', action='store_true')
    parser.add_argument('--use_adaptive_mini_batch', action='store_true')
    parser.add_argument('--use_per_sample_loss_normalization', action='store_true')
    parser.add_argument('--use_bos_token', action='store_true')

    parser.add_argument('--use_tqdm', action='store_true')
    parser.add_argument('--disable_wandb', action='store_true')

    parser.add_argument('--seed', metavar='SEED', type=int, default=99)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    output_dir = os.path.join(args.output_dir, args.run_name)
    if os.path.exists(output_dir):
        raise ValueError('output directory (%s) already exists' % str(output_dir))
    os.mkdir(output_dir)

    log_file = os.path.join(output_dir, 'info.log')
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] (%(asctime)s) %(message)s',
        datefmt='%H:%M:%S %d/%m/%Y',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    set_seed(args.seed)

    logging.info('load arc tasks')
    train_tasks, val_tasks, test_tasks = prepare_arc_tasks('data/arc1')

    logging.info('\tnum train tasks: %d' % len(train_tasks))
    logging.info('\tnum val tasks: %d' % len(val_tasks))
    logging.info('\tnum test tasks: %d' % len(test_tasks))

    logging.info('create val and test datasets from arc tasks')
    val_in_dataset = ARCDataset(train_tasks)
    val_out_dataset = ARCDataset(val_tasks)
    test_dataset = ARCDataset(test_tasks)

    logging.info('\tnum val/in samples: %d' % len(val_in_dataset))
    logging.info('\tnum val/out samples: %d' % len(val_out_dataset))
    logging.info('\tnum test samples: %d' % len(test_dataset))

    logging.info('load arc2 tasks')
    arc2_train_tasks, arc2_val_tasks, arc2_test_tasks = prepare_arc_tasks('data/arc2')

    logging.info('\tnum train tasks arc2: %d' % len(arc2_train_tasks))
    logging.info('\tnum val tasks arc2: %d' % len(arc2_val_tasks))
    logging.info('\tnum test tasks arc2: %d' % len(arc2_test_tasks))

    logging.info('create val and test datasets from arc2 tasks')
    arc2_train_dataset = ARCDataset(arc2_train_tasks)
    arc2_val_dataset = ARCDataset(arc2_val_tasks)
    arc2_test_dataset = ARCDataset(arc2_test_tasks)

    logging.info('\tnum arc2 train samples: %d' % len(arc2_train_dataset))
    logging.info('\tnum arc2 val samples: %d' % len(arc2_val_dataset))
    logging.info('\tnum arc2 test samples: %d' % len(arc2_test_dataset))

    logging.info(f'load train dataset: {args.train_dataset}')

    if args.num_steps_per_epoch is None:
        num_samples = None
    else:
        num_samples = args.num_steps_per_epoch * args.train_batch_size * args.gradient_accumulation_steps
    train_dataset = HuggingFaceDataset(args.train_dataset, num_samples=num_samples, seed=args.seed)
    logging.info('\tnum train samples: %d' % len(train_dataset))

    logging.info('load tokenizer')
    tokenizer = TaskTokenizer(
        add_row_sep=args.add_row_sep,
        segregate=args.segregate,
        use_bos_token=args.use_bos_token,
        max_position_idx=args.max_position_embeddings,
        target_only_test_output=args.target_only_test_output,
        memory_tokens_strategy=args.memory_tokens_strategy,
        num_memory_tokens=args.num_memory_tokens,
    )

    logging.info('get preprocessed datasets')

    all_val_datasets = {
        'in': val_in_dataset,
        'out': val_out_dataset,
        'train2': arc2_train_dataset,
        'val2': arc2_val_dataset,
    }

    train_dataset, all_val_datasets = get_tokenized_datasets(train_dataset, all_val_datasets, tokenizer)

    if args.pretrained_path_hf is None:
        logging.info('loading raw %s model' % args.model_type)
        model = get_raw_gpt2(
            tokenizer,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            max_position_embeddings=args.max_position_embeddings,
        )
    else:
        assert False
        logging.info('loading model from %s' % args.pretrained_path_hf)
        ...

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.from_checkpoint is None:
        last_iter = -1
    else:
        logging.info('loading from checkpoint %s' % args.from_checkpoint)
        ckpt_dict = torch.load(args.from_checkpoint, weights_only=False)
        last_iter = ckpt_dict['iter']
        model_state_dict = ckpt_dict['model_state_dict']
        optimizer_state_dict = ckpt_dict.get('optimizer_state_dict', None)

        logging.info('\tloading model state dict')
        model.load_state_dict(model_state_dict)

        if optimizer_state_dict is not None:
            logging.info('\tloading optimizer state dict')
            optimizer_state_dict['param_groups'][0]['lr'] = args.learning_rate
            optimizer_state_dict['param_groups'][0]['weight_decay'] = args.weight_decay
            optimizer.load_state_dict(optimizer_state_dict)

    logging.info('\tnum parameters in loaded model: %d' % count_trainable_parameters(model))

    logging.info('setup wandb for progress tracking')

    model_type = model.config.model_type
    run_name = f'{model_type}-{args.run_name}'

    run_cfg = dict(model_type=model_type) | vars(args)

    if not args.disable_wandb:
        wandb.init(
            project='ARC',
            name=run_name,
            config=run_cfg,
        )

    config_file = os.path.join(output_dir, 'config.json')
    logging.info('save run config to %s' % config_file)
    with open(config_file, 'w') as f:
        json.dump(run_cfg, f, indent=2)

    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        lr_scheduler_type=args.scheduler_type,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        max_grad_norm=1.0,
        max_steps=args.num_steps,
        logging_steps=args.log_every,
        eval_strategy='steps',
        eval_steps=args.eval_every,
        save_steps=args.save_every,
        bf16=args.use_bf16,
        report_to='none' if args.disable_wandb else 'wandb',
        disable_tqdm=not args.use_tqdm,
        include_for_metrics=['loss'],
    )

    train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        all_val_datasets=all_val_datasets,
        training_args=training_args,
        target_only_test_output=args.target_only_test_output,
        use_per_sample_loss_normalization=args.use_per_sample_loss_normalization,
        resume_from_checkpoint=args.from_checkpoint,
    )
