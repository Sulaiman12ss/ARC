from functools import partial
import logging

import torch
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel  # type: ignore


def calculate_metrics_default(generated_boards_1d, true_board_1d, tokenizer, special_tokens_map):
    def all_equal_at(t1, t2, value):
        return torch.equal(torch.where(t1 == value)[0], torch.where(t2 == value)[0])

    generated_boards_1d = [torch.array(board_1d) for board_1d in generated_boards_1d]
    true_board_1d = torch.array(true_board_1d)

    all_correct = any(torch.equal(board_1d, true_board_1d) for board_1d in generated_boards_1d)

    metrics = {
        'all_correct': all_correct,
    }

    special_tokens_correct = {}
    for key, token in special_tokens_map.items():
        special_tokens_correct[f'all_{key}_correct'] = any(
            all_equal_at(board_1d, true_board_1d, token) for board_1d in generated_boards_1d
        )
    metrics['special_tokens_correct'] = special_tokens_correct

    colors_correct = {}
    for color_id in range(10):
        if not (true_board_1d == color_id).any():
            continue
        colors_correct[tokenizer.decode([color_id])[0]] = any(
            all_equal_at(board_1d, true_board_1d, color_id) for board_1d in generated_boards_1d
        )
    metrics['colors_correct'] = colors_correct

    return metrics


class GPT2CustomPositions(GPT2LMHeadModel):
    @torch.inference_mode()
    def prepare_inputs_for_generation(
        self, input_ids, position_ids_prefix, num_train_boards, row_sep_token_id, **kwargs,
    ):
        prepared_inputs = GPT2LMHeadModel.prepare_inputs_for_generation(self, input_ids, **kwargs)
        t = prepared_inputs.pop('position_ids')
        bs = t.shape[0]

        prefix_seq_len = position_ids_prefix.shape[1]
        now_seq_len = input_ids.shape[1]
        n_generated = now_seq_len - prefix_seq_len

        if n_generated == 0:
            position_ids = position_ids_prefix.expand(bs, -1)
        else:
            # with for loop, for some samples might (?) be over the EOS token and contain gibberish
            cum_id_stack = []
            for i in range(bs):
                cum_id = 0
                num_rows_past = 0

                for j in range(n_generated - 1):
                    if (input_ids[i, prefix_seq_len + j] == row_sep_token_id).all():
                        num_rows_past += 1
                        cum_id = num_rows_past * 32
                    else:
                        cum_id += 1
                cum_id_stack.append(cum_id)

            cum_id_ten = torch.as_tensor(cum_id_stack, device=input_ids.device, dtype=torch.long)

            position_ids = (2 * num_train_boards + 1) * 1024 + cum_id_ten.unsqueeze(1)

        position_ids = position_ids.contiguous()

        prepared_inputs['position_ids'] = position_ids
        prepared_inputs['position_ids_prefix'] = position_ids_prefix
        prepared_inputs['num_train_boards'] = num_train_boards
        prepared_inputs['row_sep_token_id'] = row_sep_token_id

        return prepared_inputs


def convert_to_custom_gpt2(model: GPT2LMHeadModel):
    logger = logging.getLogger()
    logger.info('`convert_to_custom_gpt2` call siezes the ownership of `model` and changes it`s type in-place')
    model.__class__ = GPT2CustomPositions
    return model


@torch.inference_mode()
def do_generate(model, loader, tokenizer, max_new_tokens, calculate_metrics_fn=None, **generate_kwargs):
    if loader.batch_size != 1:
        raise ValueError('batched generation is not yet supported')

    if not isinstance(model, GPT2CustomPositions):
        assert isinstance(model, GPT2LMHeadModel), 'only gpt2 currently supported'
        model = convert_to_custom_gpt2(model)

    if calculate_metrics_fn == 'default':
        special_tokens_map = {
            'eos': tokenizer.sample_sep_token_id,
            'row_sep': tokenizer.row_sep_token_id,
        }
        calculate_metrics_fn = partial(
            calculate_metrics_default,
            tokenizer=tokenizer,
            special_tokens_map=special_tokens_map,
        )

    all_statistics = []

    device = next(model.parameters()).device
    for batch in tqdm(loader):
        num_train_boards = len(batch['augmented_sample'][0]['train'])

        input_ids = batch['input_ids'].to(device)
        position_ids = batch['position_ids'].to(device)

        labels = batch.get('labels', None)

        if labels is None:
            input_ids_prefix = input_ids
            position_ids_prefix = position_ids
        else:
            labels = labels.to(device)
            input_ids_prefix = torch.masked_select(input_ids.squeeze(0), labels.squeeze(0) == -100).unsqueeze(0)
            position_ids_prefix = torch.masked_select(position_ids.squeeze(0), labels.squeeze(0) == -100).unsqueeze(0)

        out = model.generate(
            input_ids=input_ids_prefix,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.sample_sep_token_id,
            # GPT2CustomPositions kwargs
            position_ids_prefix=position_ids_prefix,
            num_train_boards=num_train_boards,
            row_sep_token_id=tokenizer.row_sep_token_id,
            # specify generation strategy
            **generate_kwargs,
        )

        start_seq_len = input_ids_prefix.shape[1]

        generated_boards_1d = [out[i, start_seq_len:] for i in range(len(out))]

        true_board_1d = None
        if labels is not None:
            true_board_1d = torch.masked_select(labels.squeeze(0), labels.squeeze(0) != -100)

        metrics = None
        if (true_board_1d is not None) and (calculate_metrics_fn is not None):
            metrics = calculate_metrics_fn(generated_boards_1d, true_board_1d)

        generation_result = dict()

        generation_result['key'] = batch['key'][0]
        generation_result['generated_boards_1d'] = [board_1d.cpu().numpy() for board_1d in generated_boards_1d]
        generation_result['true_board_1d'] = true_board_1d.cpu().numpy() if true_board_1d is not None else None
        generation_result['metrics'] = metrics

        all_statistics.append(generation_result)

    return all_statistics
