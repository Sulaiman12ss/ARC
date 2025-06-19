from functools import partial

import torch
from transformers import EvalPrediction  # type: ignore
from tqdm.auto import tqdm

from utils import get_test_output_only_mask
from utils.generate import calculate_metrics_default


@torch.inference_mode()
def compute_metrics(
    eval_pred: EvalPrediction,
    target_only_test_output=False,
    use_per_sample_loss_normalization=False,
    **kwargs,
) -> dict:
    losses = torch.from_numpy(eval_pred.losses)
    logits = torch.from_numpy(eval_pred.predictions)
    labels = torch.from_numpy(eval_pred.label_ids)

    assert logits.ndim == 3, 'expected logits in `predictions` field'
    assert losses.ndim == 1, 'expected losses to be 1d'

    num_samples = labels.shape[0]
    loss_mask = (labels != -100)  # fmt: skip

    train_like_loss = losses.mean()

    shifted_labels = labels[:, 1:].contiguous()
    test_output_only_mask = loss_mask if target_only_test_output else get_test_output_only_mask(loss_mask)
    shifted_test_output_only_mask = test_output_only_mask[:, 1:]

    shifted_logits = logits[:, :-1].contiguous()
    shifted_predictions = shifted_logits.argmax(-1)

    match = torch.logical_and(shifted_predictions == shifted_labels, shifted_test_output_only_mask)
    is_task_correct = torch.logical_or(match, (~shifted_test_output_only_mask)).all(1)

    num_solved = is_task_correct.sum()
    num_correct_pixels = match.sum()
    num_test_pixels = shifted_test_output_only_mask.sum()

    if not target_only_test_output:
        loss_raw = torch.nn.functional.cross_entropy(
            shifted_logits.view(-1, shifted_logits.shape[-1]),
            shifted_labels.view(-1),
            reduction='none',
        )
        loss_raw = loss_raw.view(num_samples, -1)
        loss_raw = loss_raw * shifted_test_output_only_mask
        if use_per_sample_loss_normalization:
            test_output_only_loss = (loss_raw.sum(1) / shifted_test_output_only_mask.sum(1)).mean()
        else:
            test_output_only_loss = loss_raw.mean()
    else:
        test_output_only_loss = train_like_loss

    res_metrics = dict()

    res_metrics['train_like_loss'] = train_like_loss
    res_metrics['test_output_only_loss'] = test_output_only_loss
    res_metrics['per_pixel_acc'] = num_correct_pixels / num_test_pixels
    res_metrics['per_task_acc'] = num_solved / num_samples

    res_metrics['correct_ids'] = torch.nonzero(is_task_correct).squeeze(1).tolist()

    color_freqs = []
    for i in range(num_samples):
        predicted_board = shifted_predictions[i, shifted_test_output_only_mask[i]]
        predicted_colors = torch.bincount(predicted_board, minlength=10)[:10]
        color_freqs.append(predicted_colors)
    color_freqs = torch.stack(color_freqs)

    color_freqs = color_freqs.float()
    color_freqs = color_freqs / color_freqs.sum(1, keepdim=True).clip(min=1)

    avg_color_freqs = color_freqs.mean(0)
    avg_color_entropy = -torch.sum(color_freqs * torch.log(color_freqs.clip(min=1e-6)), 1).mean()
    res_metrics['avg_color_freqs'] = avg_color_freqs.tolist()
    res_metrics['avg_color_entropy'] = avg_color_entropy

    return res_metrics


@torch.inference_mode()
def evaluate(model, val_loader, tokenizer, calculate_metrics_fn=None):
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
    for i, batch in enumerate(tqdm(val_loader)):
        input_ids = batch['input_ids'].to(device)
        position_ids = batch['position_ids'].to(device)
        labels = batch['labels'].to(device)

        out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            # labels=labels,
            attention_mask=(input_ids != tokenizer.pad_token_id),
            output_attentions=False,
            output_hidden_states=False,
        )

        shifted_labels = labels[:, 1:]
        shifted_test_output_mask = (shifted_labels != -100)  # fmt: skip
        shifted_predictions = out.logits[:, :-1, :].argmax(-1)
        del out

        for j in range(labels.shape[0]):
            true_board_1d = torch.masked_select(shifted_labels[j], shifted_test_output_mask[j])
            generated_board_1d = torch.masked_select(shifted_predictions[j], shifted_test_output_mask[j])

            metrics = None
            if calculate_metrics_fn is not None:
                metrics = calculate_metrics_fn([generated_board_1d], true_board_1d)

            evaluate_results = dict()
            evaluate_results['key'] = batch['key'][j]
            evaluate_results['true_board_1d'] = true_board_1d.cpu().numpy()
            evaluate_results['generated_board_1d'] = generated_board_1d.cpu().numpy()
            evaluate_results['metrics'] = metrics

        all_statistics.append(evaluate_results)

    return all_statistics
