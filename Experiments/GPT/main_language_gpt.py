#!/usr/bin/env python3
import os
import sys
import argparse
import json
import pandas as pd
from pathlib import Path
from openai import OpenAI

TEST_TYPE = "2_Lang_{lang}"

SYSTEM_PROMPT = "Give me only a python script as a response, with only a single chart."

USER_PROMPT_TEMPLATES = {
    "eng": """Write a python script that generates a chart using CSV file {csv_filename}.

CSV file metadata:
{csv_meta}

Full CSV data:
{csv_full_content}
""",
    "kor": """CSV 파일 {csv_filename}을 사용하여 하나의 차트를 생성하는 파이썬 스크립트를 작성하라.

CSV 파일 메타데이터:
{csv_meta}

CSV 전체 데이터:
{csv_full_content}
""",
    "jap": """CSV ファイル {csv_filename} を使用して、1つのグラフを生成する Python スクリプトを書いてください。

CSV ファイルのメタデータ：
{csv_meta}

CSV の全データ：
{csv_full_content}
""",
    "chi": """使用 CSV 文件 {csv_filename} 编写一个生成单个图表的 Python 脚本。

CSV 文件元数据：
{csv_meta}

CSV 完整数据：
{csv_full_content}
""",
    "esp": """Escribe un script en Python que genere un gráfico utilizando el archivo CSV {csv_filename}.

Metadatos del archivo CSV:
{csv_meta}

Datos completos del CSV:
{csv_full_content}
""",
}

def save_python_script(code: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(code.rstrip() + "\n")

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send a CSV-to-chart prompt to Claude and save the returned Python script."
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
        "--lang",
        choices=["eng", "kor", "jap", "chi", "esp"],
        required=True,
        help="Prompt language",
    )

    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        return 2

    df = pd.read_csv(Path(f"dataset/{args.csv_path}"))

    csv_full_content = df.to_string()

    csv_meta = f"""
    Columns: {list(df.columns)}
    Shape: {df.shape[0]} rows, {df.shape[1]} columns
    Data types: {df.dtypes.to_dict()}
    """

    user_prompt = USER_PROMPT_TEMPLATES[args.lang].format(
        csv_filename=args.csv_path,
        csv_meta=csv_meta,
        csv_full_content=csv_full_content,
    )

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
    out_file = Path(
        f"output/{TEST_TYPE.format(lang=args.lang)}_{csv_filename}_generated_chart.py"
    )
    out_script = response.output_text.strip()
    save_python_script(out_script, out_file)

    response_data = {
        "input_csv_file": str(args.csv_path),
        "model_info": args.model,
        "test_type": TEST_TYPE.format(lang=args.lang),
        "language": args.lang,
        "response": out_script,
    }

    with open("claude_responses.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(response_data, ensure_ascii=False) + "\n")

    print(f"Saved generated script to {out_file.resolve()}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())