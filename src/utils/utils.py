import torch


def get_test_output_only_mask(loss_mask: torch.Tensor) -> torch.Tensor:
    loss_mask = loss_mask.long()
    batch_size, seq_len = loss_mask.shape

    x1 = loss_mask.diff(1).flip(1)
    i_start_flipped = x1.argmax(1)
    i_start = seq_len - i_start_flipped - 1

    x2 = loss_mask.flip(1).diff(1, prepend=torch.zeros(batch_size, 1, device=loss_mask.device, dtype=loss_mask.dtype))
    i_end_flipped = x2.argmax(1)
    i_end = seq_len - i_end_flipped

    p = torch.arange(seq_len, device=loss_mask.device).unsqueeze(0)

    ones_mask = torch.logical_and(i_start.unsqueeze(1) <= p, p < i_end.unsqueeze(1))
    return ones_mask


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_weights_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total_norm += p.data.norm(2).item() ** 2
    return total_norm**0.5


def load_state_dict_after_bos_token_added(model, old_ckpt_path):
    ckpt_dict = torch.load(old_ckpt_path, weights_only=True)
    # now we adjust the parameters to accept new indexation after `bos_token` was injected in between somewhere

    # token embeddings only need to be adjusted to new shape, while no inputs with `bos_token` are expected
    # so the actual weight for `bos_token` does not play any role
    wte_weight = ckpt_dict['model_state_dict']['transformer.wte.weight']
    wte_weight_with_idle_bos_weight = torch.cat([
        wte_weight[:tokenizer.bos_token_id],
        torch.zeros(1, wte_weight.shape[1], dtype=wte_weight.dtype, device=wte_weight.device),
        wte_weight[tokenizer.bos_token_id:],
    ], dim=0)

    # we adjust the head to assign zero probability to `bos_token` on generation
    # to achieve it, we introduce bias term, which is zero for all the tokens before
    # and large negative number for `bos_token`

    # 1. we zero out the effect of `bos_token` embeddings
    lm_head_weight = ckpt_dict['model_state_dict']['lm_head.weight']
    lm_head_weight_with_idle_bos_weight = torch.cat([
        lm_head_weight[:tokenizer.bos_token_id],
        torch.zeros(1, lm_head_weight.shape[1], dtype=lm_head_weight.dtype, device=lm_head_weight.device),
        lm_head_weight[tokenizer.bos_token_id:],
    ], dim=0)

    ckpt_dict['model_state_dict']['transformer.wte.weight'] = wte_weight_with_idle_bos_weight
    ckpt_dict['model_state_dict']['lm_head.weight'] = lm_head_weight_with_idle_bos_weight

    model.load_state_dict(ckpt_dict['model_state_dict'])

    # we create a bias to vanish the effect of `bos_token` for model output
    assert model.lm_head.bias is None

    old_lm_head = model.lm_head
    new_lm_head = torch.nn.Linear(
        old_lm_head.in_features,
        old_lm_head.out_features,
        bias=True,
    )

    # copy weights
    new_lm_head.weight.data = old_lm_head.weight.data.clone()

    # vanish out `bos_token`
    new_lm_head.bias.data.zero_()
    new_lm_head.bias.data[tokenizer.bos_token_id] = 1e-9

    # make bias non-trainable
    new_lm_head.bias.requires_grad = False

    # replace the old head with the new one
    model.lm_head = new_lm_head

    return model
