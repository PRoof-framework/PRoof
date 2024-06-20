import itertools
import argparse
import json
from typing import List

def get(obj: object, key: list) -> object:
    if not key:
        return obj
    if key[0] == '*':
        loop_value = obj.values() if isinstance(obj, dict) else obj
        result = []
        for x in loop_value:
            d = get(x, key[1:])
            if isinstance(d, list):
                result += d
            else:
                result.append(d)
        return result
    return get(obj[key[0]], key[1:])


def get_data(_report_files: List[str], _key: str, _escape=True) -> str:
    results_str = []
    keys = list(map(str.strip, _key.split(';')))
    results_str.append(','.join(keys))

    for name in _report_files:
        with open(name) as f:
            data = json.load(f)
            results = []
            for key in keys:
                key = key.split('.')
                try:
                    result = get(data, key)
                except Exception as e:
                    result = "not found"
                if not isinstance(result, list):
                    result = [result]
                results.append(result)

            for line in itertools.zip_longest(*results):
                if _escape:
                    results_str.append(', '.join(str(x).replace(',', '_') for x in line))
                else:
                    results_str.append(', '.join(str(x) for x in line))
    return '\n'.join(results_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = "PRoof report value reader",
        description = "dump values from report file as CSV table",
        epilog = "WIP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("key", help="key, e.g. 'model.bench.results.*.layer_prof.*.name; model.bench.results.*.layer_prof.*.flops; model.bench.results.*.layer_prof.*.memory'")
    parser.add_argument("-e", "--escape", action="store_true", help="replace ',' to '_' in data unit")
    parser.add_argument("report_files", nargs='+', help="e.g. report1.json report2.json ...")
    args = parser.parse_args()
    print(get_data(args.report_files, args.key, args.escape))
