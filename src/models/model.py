import torch
from peft import LoraConfig, get_peft_model
from transformers import BertConfig, BertForMaskedLM, GPT2Config, GPT2LMHeadModel  # type: ignore


def _update_position_embeddings_layer(bert_model, max_position_embeddings):
    # bert_model.resize_position_embeddings(max_position_embeddings)

    bert_model.config.max_position_embeddings = max_position_embeddings

    _, pos_emb_dim = bert_model.bert.embeddings.position_embeddings.weight.data.shape

    with torch.no_grad():
        new_position_embeddings_w = torch.nn.Embedding(max_position_embeddings, pos_emb_dim)
        bert_model.bert.embeddings.position_embeddings = new_position_embeddings_w

    bert_model.bert.embeddings.position_embeddings.weight.requires_grad = True

    return bert_model


def _update_token_embeddings_layer(bert_model, tokenizer):
    with torch.no_grad():
        bert_model.resize_token_embeddings(len(tokenizer.bert_tokenizer), mean_resizing=False)

        ids_span = tokenizer.all_token_ids
        ids_span_embs_w = bert_model.bert.embeddings.word_embeddings.weight[ids_span].clone()

        bert_model.resize_token_embeddings(len(ids_span_embs_w))
        bert_model.bert.embeddings.word_embeddings.weight.copy_(ids_span_embs_w)

        # reset pad token id
        bert_model.bert.embeddings.word_embeddings.padding_idx = tokenizer.pad_token_id

        # new_token_embeddings_w = torch.nn.Embedding(*ids_span_embs.shape, padding_idx=tokenizer.pad_token_id)
        # new_token_embeddings_w.weight.data.copy_(ids_span_embs)

        # bert_model.bert.embeddings.word_embeddings = new_token_embeddings_w

    bert_model.bert.embeddings.word_embeddings.weight.requires_grad = True

    # map model tokens to new id
    for config in [bert_model.config, bert_model.generation_config]:
        for k, v in list(config.to_dict().items()):
            if k.endswith('token_id'):
                setattr(config, k, getattr(tokenizer, k, None))

    return bert_model


def fit_bert_model_to_tokenizer(bert_model, tokenizer, max_position_embeddings=None):
    if max_position_embeddings is not None:
        bert_model = _update_position_embeddings_layer(bert_model, max_position_embeddings)
    bert_model = _update_token_embeddings_layer(bert_model, tokenizer)
    return bert_model


def get_raw_bert(
    tokenizer,
    hidden_size=512,
    num_hidden_layers=10,
    num_attention_heads=8,
    max_position_embeddings=32000,
):
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=tokenizer.pad_token_id,
        # intermediate_size=3072,
    )

    bert_model = BertForMaskedLM(config=config)
    return bert_model


def get_pretrained_bert(pretrained_path, tokenizer, max_position_embeddings):
    bert_model = BertForMaskedLM.from_pretrained(pretrained_path)
    bert_model = fit_bert_model_to_tokenizer(bert_model, tokenizer, max_position_embeddings)
    return bert_model


def get_raw_gpt2(
    tokenizer,
    hidden_size=256,
    num_hidden_layers=8,
    num_attention_heads=8,
    max_position_embeddings=16000,
):
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_embd=hidden_size,
        n_layer=num_hidden_layers,
        n_head=num_attention_heads,
        n_positions=max_position_embeddings,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=getattr(tokenizer, 'bos_token_id', 50256),
        eos_token_id=getattr(tokenizer, 'eos_token_id', 50256),
    )

    gpt2_model = GPT2LMHeadModel(config=config)
    return gpt2_model


def _update_gpt2_position_embeddings_layer(gpt2_model, max_position_embeddings):
    gpt2_model.config.max_position_embeddings = max_position_embeddings

    _, pos_emb_dim = gpt2_model.transformer.wpe.weight.data.shape

    with torch.no_grad():
        new_position_embeddings_w = torch.nn.Embedding(max_position_embeddings, pos_emb_dim)
        gpt2_model.transformer.wpe = new_position_embeddings_w

    gpt2_model.transformer.wpe.weight.requires_grad = True

    return gpt2_model


def _update_gpt2_token_embeddings_layer(gpt2_model, tokenizer):
    with torch.no_grad():
        gpt2_model.resize_token_embeddings(len(tokenizer.gpt2_tokenizer), mean_resizing=False)

        ids_span = tokenizer.all_token_ids
        ids_span_embs_w = gpt2_model.transformer.wte.weight[ids_span].clone()

        gpt2_model.resize_token_embeddings(len(ids_span_embs_w))
        gpt2_model.transformer.wte.weight.copy_(ids_span_embs_w)

        # reset pad token id
        gpt2_model.transformer.wte.padding_idx = tokenizer.pad_token_id

    gpt2_model.transformer.wte.weight.requires_grad = True

    # map model tokens to new id
    for config in [gpt2_model.config, gpt2_model.generation_config]:
        for k, v in list(config.to_dict().items()):
            if k.endswith('token_id'):
                setattr(config, k, getattr(tokenizer, k, None))

    return gpt2_model


def fit_gpt2_model_to_tokenizer(gpt2_model, tokenizer, max_position_embeddings=None):
    if max_position_embeddings is not None:
        gpt2_model = _update_gpt2_position_embeddings_layer(gpt2_model, max_position_embeddings)
    gpt2_model = _update_gpt2_token_embeddings_layer(gpt2_model, tokenizer)
    return gpt2_model


def get_pretrained_gpt2_w_lora(
    pretrained_path,
    tokenizer,
    max_position_embeddings=None,
):
    model = GPT2LMHeadModel.from_pretrained(pretrained_path)
    lora_config = LoraConfig(
        r=8,  # Rank of the low-rank matrices
        lora_alpha=16,  # Scaling factor
        target_modules=['q_proj', 'v_proj'],  # Layers to apply LoRA to
        lora_dropout=0.0,
        bias='none',
    )

    model = fit_gpt2_model_to_tokenizer(model, tokenizer, max_position_embeddings=max_position_embeddings)

    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if 'lora' in name or 'adapter' in name:
            param.requires_grad = True  # Ensure LoRA layers are trainable
        else:
            param.requires_grad = False  # Freeze other parameters

    return model
