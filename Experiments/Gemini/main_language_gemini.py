#!/usr/bin/env python3
import os
import sys
import argparse
import json
import time
import pandas as pd
from pathlib import Path
import google.generativeai as genai
from google.api_core import exceptions

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
        description="Send a CSV-to-chart prompt to Gemini (Language Exp) and save the returned Python script."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="myfile.csv",
        help="Path to the CSV file (default: myfile.csv)",
    )
    parser.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Google Gemini model name (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--lang",
        choices=["eng", "kor", "jap", "chi", "esp"],
        required=True,
        help="Prompt language",
    )

    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        return 2
    
    genai.configure(api_key=api_key)

    if args.csv_path.startswith("dataset/"):
        csv_full_path = Path(args.csv_path)
    else:
        csv_full_path = Path(f"dataset/{args.csv_path}")

    try:
        df = pd.read_csv(csv_full_path)
    except FileNotFoundError:
        print(f"ERROR: File not found at {csv_full_path}", file=sys.stderr)
        return 1

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

    model = genai.GenerativeModel(
        model_name=args.model,
        system_instruction=SYSTEM_PROMPT
    )

    max_retries = 3
    out_script = ""
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                user_prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.0)
            )
            out_script = response.text.strip()
            break 
            
        except exceptions.ResourceExhausted:
            print(f"⚠️ Rate limit hit! Waiting 30 seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(30)
        except Exception as e:
            print(f"Error generating content: {e}")
            return 1
    
    if not out_script:
        print("ERROR: Failed to generate script after retries.")
        return 1

    if out_script.startswith("```"):
        out_script = out_script.replace("```python", "").replace("```", "").strip()

    csv_filename = Path(args.csv_path).stem
    formatted_test_type = TEST_TYPE.format(lang=args.lang)
    out_file = Path(f"output/{formatted_test_type}_{csv_filename}_generated_chart.py")
    
    save_python_script(out_script, out_file)

    response_data = {
        "input_csv_file": str(args.csv_path),
        "model_info": args.model,
        "test_type": formatted_test_type,
        "language": args.lang,
        "response": out_script,
    }

    with open("gemini_responses.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(response_data, ensure_ascii=False) + "\n")

    print(f"Saved generated script to {out_file.resolve()}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())