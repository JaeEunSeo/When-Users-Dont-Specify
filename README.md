# When Users Don’t Specify: How LLMs Make Design Decisions in Underspecified Data Visualization Contexts
![arXiv: preprint](https://img.shields.io/badge/arXiv-preprint-orange.svg)

This is the official repository for the paper [When Users Don’t Specify: How LLMs Make Design Decisions in Underspecified Data Visualization Contexts](). <br>
```
# Directory Structure
├── Charts
│   └── *.png
├── Data_Analysis
│   └── analysis_tool.py
├── Dataset
├── Experiments
│   ├── Claude
│   │   ├── main_claude.py
│   │   ├── main_expertise_claude.py
│   │   ├── main_language_claude.py
│   │   └── main_literacy_claude.py
│   ├── Gemini
│   │   ├── main_expertise_gemini.py
│   │   ├── main_gemini.py
│   │   ├── main_language_gemini.py
│   │   └── main_literacy_gemini.py
│   └── GPT
│       ├── main_expertise_gpt.py
│       ├── main_gpt.py
│       ├── main_language_gpt.py
│       └── main_literacy_gpt.py
├── LLM_Judgement
│   ├── claude_responses_judged.jsonl
│   ├── gemini_responses_judged.jsonl
│   └── openai_responses_judged.jsonl
├── Output_json
│   ├── claude_output.jsonl
│   ├── gemini_responses.jsonl
│   └── openai_responses.jsonl
├── README.md
└── requirements.txt
```

Chart image files are not included in the repository due to size constraints. For consistency, please download them from [Google Drive](https://drive.google.com/drive/folders/1XG0LFgSTGtiSn30ftr6hqGoCNyEzYgD4?usp=drive_link) and place them in the `Charts/` directory.
