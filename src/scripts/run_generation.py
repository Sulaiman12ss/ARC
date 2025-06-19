import argparse
import json

import torch
from transformers import DataCollatorWithPadding, set_seed  # type: ignore

from data.arc import prepare_arc_tasks
from data.dataset import ARCDataset, AugmentDataset
from models.model import get_raw_gpt2
from models.tokenizer import TaskTokenizer
from utils.generate import do_generate


class ExtCollator(DataCollatorWithPadding):
    def __call__(self, features):
        raw_data = {
            'key': [item['key'] for item in features],
            'augmented_key': [item['augmented_key'] for item in features],
            'raw_sample': [item['raw_sample'] for item in features],
            'augmented_sample': [item['augmented_sample'] for item in features],
        }
        collated_data = super().__call__(features)
        return raw_data | collated_data


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

    ext_collator = ExtCollator(tokenizer)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=ext_collator,
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

    all_statistics = do_generate(
        model,
        val_loader,
        tokenizer,
        max_new_tokens=tokenizer.MAX_ONE_SEQ_LEN,
        calculate_metrics_fn='default',
        # generate_kwargs
        do_sample=False,
        num_beams=args.num_beams,
        early_stopping=(args.num_beams > 1),
    )

    for gen_result in all_statistics:
        gen_result['generated_boards_1d'] = [board_1d.tolist() for board_1d in gen_result['generated_boards_1d']]
        gen_result['true_board_1d'] = gen_result['true_board_1d'].tolist()

    print(
        f'num correct predictions: {sum(any(gen_result["tags"]["is_full_correct"]) for gen_result in all_statistics)}'
    )

    with open(args.output_file, 'w') as f:
        json.dump(all_statistics, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generation with pre-trained gpt2')
    parser.add_argument('--model-ckpt-path', type=str, required=True)
    parser.add_argument('--model-config-path', type=str, required=True)
    parser.add_argument(
        '--num-beams', type=int, default=1, help='if equals 1, run greedy decoding. otherwise, run beam search'
    )
    parser.add_argument('--output-file', type=str, default='result.json', help='file to save result generation result')

    args = parser.parse_args()
    main(args)
