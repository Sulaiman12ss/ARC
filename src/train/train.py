import logging
from functools import partial

import torch
from transformers import DataCollatorWithPadding, Trainer, TrainerCallback  # type: ignore

from train.evaluate import compute_metrics
from utils import get_test_output_only_mask, get_model_weights_norm


class SolvedOnEvalCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.logger = logging.getLogger(__name__)

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        now_val_key = next(iter(metrics.keys())).split('_')[1]
        self.logger.info('step %d. after evaluation. key: val/%s' % (state.global_step, now_val_key))

        if state.is_world_process_zero:
            now_val_dataset = self.trainer.eval_dataset[now_val_key]

            correct_ids_key = f'eval_{now_val_key}_correct_ids'
            correct_ids = metrics[correct_ids_key]

            if correct_ids:
                correct_keys = [now_val_dataset[correct_id]['key'] for correct_id in correct_ids]
                self.logger.info(f'val/{now_val_key}. solved {len(correct_keys)} tasks: {", ".join(correct_keys)}')


def compute_loss_func_per_sample_normalization(model_output, labels, grad_accum_steps, **kwargs):
    batch_size = labels.shape[0]

    shifted_logits = model_output.logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    shifted_loss_mask = (shifted_labels != -100)  # fmt: skip

    loss_raw = torch.nn.functional.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        reduction='none',
    )

    loss_raw = loss_raw.view(shifted_labels.shape[0], -1)
    loss_raw = loss_raw * shifted_loss_mask

    loss = torch.sum(loss_raw.sum(1) / shifted_loss_mask.sum(1)) / (batch_size * grad_accum_steps)
    return loss


class AdunTrainer(Trainer):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cum_num_samples = 0
        self.cum_train_metrics = torch.zeros(3, device=device, dtype=torch.long)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs['labels']

        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch, **kwargs
        )

        with torch.inference_mode():
            predictions = outputs.logits[:, :-1, :].argmax(-1)  # type: ignore

            shifted_labels = labels[:, 1:]

            loss_mask = (labels != -100)  # fmt: skip
            shifted_predictions_mask = get_test_output_only_mask(loss_mask)[:, 1:]

            match = (predictions == shifted_labels).logical_and(shifted_predictions_mask)
            is_batch_correct = match.logical_or(~shifted_predictions_mask).all(1)

            self.cum_num_samples += labels.shape[0]
            self.cum_train_metrics[0] += match.long().sum()
            self.cum_train_metrics[1] += shifted_predictions_mask.long().sum()
            self.cum_train_metrics[2] += is_batch_correct.long().sum()

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        # custom behaviour only applicable for train-time logging
        is_train = ('grad_norm' in logs)  # fmt: skip

        if is_train and self.cum_num_samples > 0:
            # track model weights
            weights_norm = get_model_weights_norm(self.model)
            logs['weights_norm'] = weights_norm

            # additional metrics
            match_total, test_region_total, corrects_total = self.cum_train_metrics.tolist()

            per_pixel_acc = match_total / test_region_total
            per_task_acc = corrects_total / self.cum_num_samples

            logs['per_pixel_acc'] = per_pixel_acc
            logs['per_task_acc'] = per_task_acc

        self.cum_num_samples = 0
        self.cum_train_metrics.zero_()

        return super().log(logs, *args, **kwargs)


def train(
    model,
    tokenizer,
    train_dataset,
    all_val_datasets,
    training_args,
    target_only_test_output=False,
    use_per_sample_loss_normalization=False,
    resume_from_checkpoint=None,
):
    logging.info('start training')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info('\tusing device %s' % device)

    data_collator = DataCollatorWithPadding(tokenizer)

    compute_loss_func = None
    if use_per_sample_loss_normalization:
        compute_loss_func = partial(
            compute_loss_func_per_sample_normalization, grad_accum_steps=training_args.gradient_accumulation_steps
        )

    trainer = AdunTrainer(
        device=device,
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=all_val_datasets,
        compute_metrics=partial(
            compute_metrics,
            target_only_test_output=target_only_test_output,
            use_per_sample_loss_normalization=use_per_sample_loss_normalization,
        ),
        compute_loss_func=compute_loss_func,
    )
    trainer.add_callback(SolvedOnEvalCallback(trainer))

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
