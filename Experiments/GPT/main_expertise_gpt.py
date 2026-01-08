#!/usr/bin/env python3
import os
import sys
import argparse
import json
import pandas as pd
from pathlib import Path
from openai import OpenAI

TEST_TYPE = "2_expertise_{level}"
SYSTEM_PROMPT = "Give me only a python script as a response, with only a single chart. You are a {level} in data visualization. "
USER_PROMPT_TEMPLATE = """Write a python script that generates a chart using CSV file {csv_filename}.

CSV file metadata:
{csv_meta}

Full CSV data:
{csv_full_content}
"""

def read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8-sig") as f:
            return f.read()

def save_python_script(code: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(code.rstrip() + "\n")

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send a CSV-to-chart prompt to OpenAI and print the returned Python script."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="myfile.csv",
        help="Path to the CSV file (default: myfile.csv)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="OpenAI model name (default: gpt-5)",
    )
    parser.add_argument(
        "--level",
        choices=["L", "M", "H"],
        required=True,
        help="Assign expertise level"
    )

    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        return 2

    df = pd.read_csv(Path(f"dataset/{args.csv_path}"))

    # CSV 전체 내용을 문자열로 변환
    csv_full_content = df.to_string()

    # CSV 메타 정보
    csv_meta = f"""
    Columns: {list(df.columns)}
    Shape: {df.shape[0]} rows, {df.shape[1]} columns
    Data types: {df.dtypes.to_dict()}
    """

    system_prompt = SYSTEM_PROMPT.format(level=map(args.level, {"L":"beginner", "M":"intermediate", "H":"expert"}))
    user_prompt = USER_PROMPT_TEMPLATE.format(csv_filename=args.csv_path,
                                    csv_full_content=csv_full_content, 
                                    csv_meta=csv_meta)

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=args.model,
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    )

    csv_filename = Path(args.csv_path).stem
    out_file = Path(f"output/{TEST_TYPE.format(level=args.level)}_{csv_filename}_generated_chart.py")
    out_script = response.output_text.strip()
    save_python_script(out_script, out_file)

    response_data = {
        "input_csv_file": str(args.csv_path),
        "model_info": args.model,
        "test_type": TEST_TYPE.format(level=args.level),
        "response": out_script
    }

    with open('openai_responses.jsonl', "a", encoding="utf-8") as f:
        f.write(json.dumps(response_data, ensure_ascii=False) + "\n")

    print(f"Saved generated script to {out_file.resolve()}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())