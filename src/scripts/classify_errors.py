import argparse
import json
from collections import Counter, defaultdict
from typing import Any, Dict, List


def classify_metrics(metrics: Dict[str, Any]) -> str:
    """Return an error category name for a single metrics entry."""
    if metrics.get('all_correct'):
        return 'correct'

    categories = []
    special = metrics.get('special_tokens_correct', {})
    if special and not all(special.values()):
        categories.append('special_token_error')
    colors = metrics.get('colors_correct', {})
    if colors and any(not v for v in colors.values()):
        categories.append('color_error')
    if not categories:
        categories.append('other_error')
    return '+'.join(categories)


def process_file(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r') as f:
        return json.load(f)


def main(args: argparse.Namespace) -> None:
    counts: Counter[str] = Counter()
    groups: defaultdict[str, List[tuple[str, str]]] = defaultdict(list)

    for path in args.files:
        for item in process_file(path):
            cat = classify_metrics(item.get('metrics', {}))
            counts[cat] += 1
            groups[cat].append((item.get('key'), path))

    for cat, entries in groups.items():
        if cat == 'correct' or len(entries) < args.min_count:
            continue
        print(f'Error category: {cat} (count: {len(entries)})')
        for key, src in entries:
            print(f'  - {key} from {src}')
        print()

    print('Summary counts:')
    for cat, count in counts.items():
        print(f'  {cat}: {count}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify generation errors from ARC metrics JSON files.')
    parser.add_argument('files', nargs='+', help='JSON files produced by generation scripts.')
    parser.add_argument('--min-count', type=int, default=2, help='Only show categories with at least this many samples')
    main(parser.parse_args())
