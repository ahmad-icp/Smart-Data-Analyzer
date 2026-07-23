# Smart Data Analyzer Pro

Smart Data Analyzer Pro is a privacy-aware Streamlit workspace for understanding,
cleaning, profiling, visualizing, assessing, and publishing tabular datasets without
writing code.

## What it does

- Imports CSV, TSV, Excel, JSON/JSONL, and Parquet data.
- Profiles schema, missingness, duplicates, distributions, and correlations.
- Applies explainable cleaning operations with undo and transformation history.
- Builds filtered interactive Plotly dashboards.
- Runs descriptive statistics, regression, and hypothesis tests.
- Produces an explainable ML-readiness score and data dictionary.
- Generates polished dataset documentation with local Qwen/Ollama, Gemini, Groq, or
  a deterministic professional fallback.
- Exports a complete ZIP containing cleaned data, README, quality report, data
  dictionary, history, and a reproducibility script.

AI never edits the DataFrame directly. It writes documentation from schema and
aggregate quality metrics; raw records and example values are not sent to cloud
providers. Deterministic Pandas functions perform data changes.

## Run locally

```bash
git clone https://github.com/ahmad-icp/Smart-Data-Analyzer.git
cd Smart-Data-Analyzer/smart-data-analyzer
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Optional AI writers

No AI setup is required: the built-in writer always produces a complete data card.
For richer natural-language output, configure one or more providers.

### Local and private: Ollama + Qwen

```bash
ollama pull qwen3:4b
ollama serve
```

Optional environment variables:

```text
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:4b
```

### Gemini or Groq fallback

```text
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-3-flash-preview

GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.3-70b-versatile
```

Never commit API keys. Configure them in the deployment platform's secret manager.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Streamlit Community Cloud

Choose this repository and set the main file path to:

```text
smart-data-analyzer/app.py
```

Ollama is intended for local or self-hosted deployment. On Community Cloud, use the
built-in writer or configure a cloud provider key securely.
