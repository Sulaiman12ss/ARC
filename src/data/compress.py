def _flatten_board(board: list[list[int]]) -> list[int]:
    result = [len(board), len(board[0])]
    for row in board:
        result.extend(row)
    return result


def encode_sample(sample: dict[str, list[dict[str, list[list[int]]]]]) -> bytes:
    result = []

    for train in sample['train']:
        result.extend(_flatten_board(train['input']))
        result.extend(_flatten_board(train['output']))

    for test in sample['test']:
        result.extend(_flatten_board(test['input']))
        if 'output' in test:
            result.extend(_flatten_board(test['output']))

    return bytes(result)


def decode_sample(encoded: bytes) -> dict[str, list[dict[str, tuple[tuple[int, ...], ...]]]]:
    all_boards = []
    it = 0
    while it < len(encoded):
        num_rows = encoded[it]
        num_cols = encoded[it + 1]
        it += 2
        board = []
        for _ in range(num_rows):
            board.append(tuple(encoded[it : it + num_cols]))
            it += num_cols
        all_boards.append(tuple(board))

    assert it == len(encoded)

    num_train = (len(all_boards) - 1) // 2 * 2
    train_boards = all_boards[:num_train]
    test_boards = all_boards[num_train:]

    result = {
        'train': [],
        'test': [],
    }

    for i in range(0, len(train_boards), 2):
        result['train'].append({'input': train_boards[i], 'output': train_boards[i + 1]})

    if len(test_boards) == 1:
        result['test'].append({'input': test_boards[0]})
    elif len(test_boards) == 2:
        result['test'].append({'input': test_boards[0], 'output': test_boards[1]})
    else:
        assert False, 'invalid number of test_boards'

    return result


if __name__ == '__main__':
    import json

    with open('data/arc1/arc-agi_training_challenges.json', 'r') as f:
        dataset = json.load(f)

    sample = dataset['11852cab']

    for split in sample.values():
        for in_out in split:
            in_out['input'] = tuple(tuple(row) for row in in_out['input'])
            if 'output' not in in_out:
                continue
            in_out['output'] = tuple(tuple(row) for row in in_out['output'])

    encoded = encode_sample(sample)
    decoded = decode_sample(encoded)

    assert sample == decoded, 'sample is broken'
