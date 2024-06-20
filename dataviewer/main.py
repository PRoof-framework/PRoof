import argparse
import json
from pathlib import Path

import gen_text
import gen_html

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = "PRoof dataviewer",
        description = "PRoof dataviewer to convert report.json to plaintext or HTML5 page",
        epilog = "WIP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("report_file", help="e.g. report.json")
    parser.add_argument("output_path", nargs='?', default="tmp/report", help="output dir")
    args = parser.parse_args()

    with open(args.report_file, 'r') as f:
        collected_data = json.load(f)

    mode = gen_html

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    mode.gen_report(collected_data, args.output_path)
