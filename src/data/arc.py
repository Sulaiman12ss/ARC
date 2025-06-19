import copy
import json
import os


def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def _load_arc_data(base_path):
    training_challenges = load_json(os.path.join(base_path, 'arc-agi_training_challenges.json'))
    training_solutions = load_json(os.path.join(base_path, 'arc-agi_training_solutions.json'))

    evaluation_challenges = load_json(os.path.join(base_path, 'arc-agi_evaluation_challenges.json'))
    evaluation_solutions = load_json(os.path.join(base_path, 'arc-agi_evaluation_solutions.json'))

    test_challenges = load_json(os.path.join(base_path, 'arc-agi_test_challenges.json'))

    return (training_challenges, training_solutions), (evaluation_challenges, evaluation_solutions), test_challenges


def _merge_challenges_with_solutions(challenges, solutions):
    merge = copy.deepcopy(challenges)
    for key, task in challenges.items():
        merge[key]['test'] = []
        test_inputs = task['test']
        test_outputs = solutions[key]
        for input, output in zip(test_inputs, test_outputs):
            merge[key]['test'].append(
                {
                    'input': input['input'],
                    'output': output,
                }
            )
    return merge


# def filter_by_board_size(tasks, max_numel=100):
#     filtered_tasks = defaultdict(dict)
#     for key, task in tasks.items():
#         valid_train_task_ids = []
#         for i, sample in enumerate(task['train']):
#             if np.size(sample['input']) > max_numel:
#                 continue
#             else:
#                 valid_train_task_ids.append(i)
#         if not valid_train_task_ids:
#             continue

#         filtered_tasks[key]['train'] = [task['train'][valid_id] for valid_id in valid_train_task_ids]
#         filtered_tasks[key]['test'] = copy.deepcopy(task['test'])

#     return dict(filtered_tasks)


# def filter_out_only_small_size_tasks():
#     small_train_tasks = filter_by_board_size(train_tasks, max_numel=50)
#     small_val_tasks = filter_by_board_size(val_tasks, max_numel=50)

#     return small_train_tasks, small_val_tasks


def prepare_arc_tasks(arc_data_path):
    (training_challenges, training_solutions), (evaluation_challenges, evaluation_solutions), test_challenges = (
        _load_arc_data(arc_data_path)
    )

    train_tasks = _merge_challenges_with_solutions(training_challenges, training_solutions)
    val_tasks = _merge_challenges_with_solutions(evaluation_challenges, evaluation_solutions)
    test_tasks = test_challenges

    return train_tasks, val_tasks, test_tasks
