import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, GPT2Tokenizer  # type: ignore

# 0:black, 1:blue, 2:red, 3:green, 4:yellow, 5:gray, 6:pink, 7:orange, 8:sky, 9:brown


class SimpleTokenizer:
    def __init__(self):
        self.color_map = ['black', 'blue', 'red', 'green', 'yellow', 'gray', 'pink', 'orange', 'sky', 'brown']

        self.pixel_tokens = list(range(10))

        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'
        self.in_out_sep_token = '[:]'
        self.row_sep_token = '[-]'
        self.sample_sep_token = '[SEP]'
        self.pad_token = '[PAD]'
        self.mem_token = '[MEM]'
        self.special_tokens = [
            self.bos_token,
            self.eos_token,
            self.in_out_sep_token,
            self.row_sep_token,
            self.sample_sep_token,
            self.pad_token,
            self.mem_token,
        ]

        self.token_to_id = {token: i for i, token in enumerate(self.pixel_tokens + self.special_tokens)}

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        self.bos_token_id = self.token_to_id[self.bos_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.in_out_sep_token_id = self.token_to_id[self.in_out_sep_token]
        self.row_sep_token_id = self.token_to_id[self.row_sep_token]
        self.sample_sep_token_id = self.token_to_id[self.sample_sep_token]
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.mem_token_id = self.token_to_id[self.mem_token]

    def encode(self, seq):
        return [self.token_to_id[token] for token in seq]

    def decode(self, ids):
        tokens = [self.id_to_token[i] for i in ids]
        tokens = [(self.color_map[t] if t in self.pixel_tokens else t) for t in tokens]
        return tokens

    def __len__(self):
        return len(self.token_to_id)


class TaskTokenizer(SimpleTokenizer):
    """
    - Flattens boards, inserts special tokens
    - Builds position IDs
    - Encodes sequences and labels
    """

    MAX_ONE_SEQ_LEN = 1024
    MAX_ONE_ROW_LEN = 32

    def __init__(
        self,
        add_row_sep: bool = False,
        segregate: str = 'no',
        use_bos_token: bool = False,
        max_position_idx: int = 24000,
        target_only_test_output: bool = False,
        memory_tokens_strategy: str = 'left',
        num_memory_tokens: int = 0,
    ):
        super().__init__()

        assert segregate in ['row', 'board', 'no']
        assert memory_tokens_strategy in ['left', 'right']

        self.add_row_sep = add_row_sep
        self.segregate = segregate
        self.use_bos_token = use_bos_token
        self.max_position_idx = max_position_idx
        self.target_only_test_output = target_only_test_output
        self.memory_tokens_strategy = memory_tokens_strategy
        self.num_memory_tokens = num_memory_tokens
        self.field_to_pad_value = {
            'input_ids': self.pad_token_id,
            'position_ids': 0,
            'labels': -100,
        }

    def _flatten_board(self, board):
        flat = []
        for row in board:
            flat.extend(list(row))
            if self.add_row_sep:
                flat.append(self.row_sep_token)
        return flat

    def _get_board_position_ids(self, board, board_idx, add_mem_tokens=False):
        if self.segregate == 'no':
            raise ValueError("`segregate` must be 'row' or 'board' to build position IDs.")

        pos_ids = []
        base = board_idx * self.MAX_ONE_SEQ_LEN
        n_rows, n_cols = np.shape(board)
        sep_inc = 1 if self.add_row_sep else 0

        if add_mem_tokens and self.memory_tokens_strategy == 'left':
            pos_ids.extend(range(base, base + self.num_memory_tokens))
            base += self.num_memory_tokens

        if self.segregate == 'board':
            length = n_rows * (n_cols + sep_inc) + 1
            pos_ids.extend(range(base, base + length))
        else:
            for i in range(n_rows):
                start = base + i * self.MAX_ONE_ROW_LEN
                pos_ids.extend(range(start, start + n_cols + sep_inc))
            pos_ids.append(base + n_rows * self.MAX_ONE_ROW_LEN)

        if add_mem_tokens and self.memory_tokens_strategy == 'right':
            start = pos_ids[-1] + 1
            pos_ids.extend(range(start, start + self.num_memory_tokens))

        return pos_ids

    def _process_split(self, samples, input_seq, labels, pos_ids, board_idx, is_test=False):
        """
        Process samples, masking only the first training output.
        """
        for i, sample in enumerate(samples):
            flat_in = self._flatten_board(sample['input'])

            if self.memory_tokens_strategy == 'left':
                input_seq.extend([self.mem_token] * self.num_memory_tokens)
            input_seq.extend(flat_in)
            input_seq.append(self.in_out_sep_token)
            if self.memory_tokens_strategy == 'right':
                input_seq.extend([self.mem_token] * self.num_memory_tokens)

            labels.extend([self.pad_token] * (len(flat_in) + 1 + self.num_memory_tokens))

            if self.segregate != 'no':
                pos_ids.extend(self._get_board_position_ids(sample['input'], board_idx, add_mem_tokens=True))
                board_idx += 1

            flat_out = self._flatten_board(sample['output'])
            input_seq.extend(flat_out)
            input_seq.append(self.sample_sep_token)

            if is_test or (not self.target_only_test_output and i > 0):
                labels.extend(flat_out)
                labels.append(self.sample_sep_token)
            else:
                labels.extend([self.pad_token] * (len(flat_out) + 1))

            if self.segregate != 'no':
                pos_ids.extend(self._get_board_position_ids(sample['output'], board_idx, add_mem_tokens=False))
                board_idx += 1
        return board_idx

    def tokenize(self, task):
        input_seq, labels, pos_ids = [], [], []

        if self.use_bos_token:
            input_seq += [self.bos_token]
            labels += [self.pad_token]
            pos_ids += [self.max_position_idx - 1]  # hopefully we never see this index anywhere again

        board_idx = 0
        board_idx = self._process_split(task['train'], input_seq, labels, pos_ids, board_idx, is_test=False)
        board_idx = self._process_split(task['test'], input_seq, labels, pos_ids, board_idx, is_test=True)

        input_ids = self.encode(input_seq)
        result = {'input_ids': input_ids}

        if pos_ids:
            result['position_ids'] = pos_ids

        if labels:
            label_ids = self.encode(labels)
            label_ids = [t if t != self.pad_token_id else -100 for t in label_ids]
            result['labels'] = label_ids

        return result

    def save_pretrained(self, output_dir):
        pass

    def pad(
        self,
        encoded_inputs,
        padding=True,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        return_tensors: str = 'pt',
    ):
        batch = {}
        for field, pad_value in self.field_to_pad_value.items():
            tensors = [torch.tensor(e[field], dtype=torch.long) for e in encoded_inputs]
            padded_tensor = pad_sequence(tensors, batch_first=True, padding_value=pad_value)

            batch[field] = padded_tensor

            if not pad_to_multiple_of:
                continue

            seq_len = padded_tensor.size(1)
            target_len = ((seq_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

            if target_len == seq_len:
                continue

            diff = target_len - seq_len
            batch[field] = F.pad(padded_tensor, (0, diff), value=pad_value)

        attn_mask = (batch['input_ids'] != self.pad_token_id).long()
        batch['attention_mask'] = attn_mask

        if return_tensors == 'pt':
            return batch
        else:
            return {k: v.tolist() for k, v in batch.items()}


class BertBasedTokenizer:
    def __init__(self, pretrained_path='bert-base-uncased'):
        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_path)
        self.color_map = ['black', 'blue', 'red', 'green', 'yellow', 'gray', 'pink', 'orange', 'sky', 'brown']

        self.blank_token = self.bert_tokenizer.unk_token
        self.in_out_sep_token = '[:]'
        self.row_sep_token = '[-]'
        self.sample_sep_token = self.bert_tokenizer.sep_token
        self.mask_token = self.bert_tokenizer.mask_token
        self.pad_token = self.bert_tokenizer.pad_token
        self.special_tokens = [
            self.blank_token,
            self.row_sep_token,
            self.in_out_sep_token,
            self.sample_sep_token,
            self.mask_token,
            self.pad_token,
        ]

        self.bert_tokenizer.add_tokens([self.in_out_sep_token, self.row_sep_token])
        self.all_token_ids = []
        for token in self.color_map + self.special_tokens:
            ids = self.bert_tokenizer.encode(token, add_special_tokens=False)
            assert len(ids) == 1
            token_id = ids[0]
            self.all_token_ids.append(token_id)

        self.token_to_id = dict()
        # for i, token in enumerate(self.color_map):
        #     ids = self.bert_tokenizer.encode(token, add_special_tokens=False)
        #     assert len(ids) == 1
        #     token_id = ids[0]
        #     self.token_to_id[i] = token_id

        # for token in self.special_tokens:
        #     ids = self.bert_tokenizer.encode(token, add_special_tokens=False)
        #     assert len(ids) == 1
        #     token_id = ids[0]
        #     self.token_to_id[token] = token_id

        for i, token in enumerate(self.color_map):
            self.token_to_id[i] = i

        for i, token in enumerate(self.special_tokens, start=len(self.color_map)):
            self.token_to_id[token] = i

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        self.blank_token_id = self.token_to_id[self.blank_token]
        self.in_out_sep_token_id = self.token_to_id[self.in_out_sep_token]
        self.row_sep_token_id = self.token_to_id[self.row_sep_token]
        self.sample_sep_token_id = self.token_to_id[self.sample_sep_token]
        self.mask_token_id = self.token_to_id[self.mask_token]
        self.pad_token_id = self.token_to_id[self.pad_token]

    def encode(self, seq):
        return [self.token_to_id[token] for token in seq]

    def decode(self, ids):
        tokens = [self.id_to_token[i] for i in ids]
        tokens = [(self.color_map[t] if t in range(len(self.color_map)) else t) for t in tokens]
        return tokens

    def __len__(self):
        return len(self.token_to_id)


class GPT2BasedTokenizer:
    def __init__(self, pretrained_path):
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(pretrained_path)
        self.color_map = ['black', 'blue', 'red', 'green', 'yellow', 'gray', 'pink', 'orange', 'sky', 'brown']

        self.eos_token = self.gpt2_tokenizer.eos_token
        self.in_out_sep_token = '[:]'
        self.row_sep_token = '[-]'
        self.sample_sep_token = '[SEP]'
        self.pad_token = self.gpt2_tokenizer.pad_token
        self.special_tokens = [
            self.eos_token,
            self.in_out_sep_token,
            self.row_sep_token,
            self.sample_sep_token,
            self.pad_token,
        ]

        self.gpt2_tokenizer.add_tokens([self.in_out_sep_token, self.row_sep_token, self.sample_sep_token])

        self.all_token_ids = []
        for token in self.color_map + self.special_tokens:
            ids = self.gpt2_tokenizer.encode(token, add_special_tokens=False)
            assert len(ids) == 1
            token_id = ids[0]
            self.all_token_ids.append(token_id)

        self.token_to_id = dict()

        for i, token in enumerate(self.color_map):
            self.token_to_id[i] = i

        for i, token in enumerate(self.special_tokens, start=len(self.color_map)):
            self.token_to_id[token] = i

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        self.eos_token_id = self.token_to_id[self.eos_token]
        self.in_out_sep_token_id = self.token_to_id[self.in_out_sep_token]
        self.row_sep_token_id = self.token_to_id[self.row_sep_token]
        self.sample_sep_token_id = self.token_to_id[self.sample_sep_token]
        self.pad_token_id = self.token_to_id[self.pad_token]

    def encode(self, seq):
        return [self.token_to_id[token] for token in seq]

    def decode(self, ids):
        tokens = [self.id_to_token[i] for i in ids]
        tokens = [(self.color_map[t] if t in range(len(self.color_map)) else t) for t in tokens]
        return tokens

    def __len__(self):
        return len(self.token_to_id)
