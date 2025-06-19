import argparse
import json

import torch
from transformers import DataCollatorWithPadding, set_seed  # type: ignore

from data.arc import prepare_arc_tasks
from data.dataset import ARCDataset, AugmentDataset
from models.model import get_raw_gpt2
from models.tokenizer import TaskTokenizer
from train.evaluate import evaluate


def main(args):
    set_seed(99)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('use device %s' % str(device))

    _, val_tasks, _ = prepare_arc_tasks('data/arc1')

    with open(args.model_config_path, 'r') as f:
        model_config = json.load(f)

    tokenizer = TaskTokenizer(
        add_row_sep=True,
        segregate='row',
        target_only_test_output=True,
        use_bos_token=model_config.get('use_bos_token', False),
        max_position_idx=model_config.get('max_position_embeddings', 24000),
    )

    val_dataset = ARCDataset(val_tasks)
    val_dataset = AugmentDataset(val_dataset, tokenizer, augment=False)

    collator = DataCollatorWithPadding(tokenizer)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    model = get_raw_gpt2(
        tokenizer,
        hidden_size=model_config['hidden_size'],
        num_hidden_layers=model_config['num_hidden_layers'],
        num_attention_heads=model_config['num_attention_heads'],
        max_position_embeddings=model_config['max_position_embeddings'],
    )
    model.load_state_dict(torch.load(args.model_ckpt_path, weights_only=True)['model_state_dict'])
    model = model.eval().to(device)

    all_metrics = evaluate(model, val_loader, tokenizer, calculate_metrics_fn='default')

    print(f'num correct predictions: {all_metrics["is_correct"].sum().item()}')

    for key in list(all_metrics.keys()):
        value = all_metrics[key]

        if torch.is_tensor(value):
            all_metrics[key] = value.tolist()
        else:
            all_metrics[key] = [subv.tolist() for subv in value]

    with open(args.output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate pre-trained gpt2')
    parser.add_argument('--model-ckpt-path', type=str, required=True)
    parser.add_argument('--model-config-path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output-file', type=str, default='result.json', help='file to save result generation result')

    args = parser.parse_args()
    main(args)
