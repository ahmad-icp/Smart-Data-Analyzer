# Smart Data Analyzer Pro

Smart Data Analyzer Pro is a privacy-aware Streamlit workspace for understanding,
cleaning, profiling, visualizing, assessing, and publishing tabular datasets without
writing code.

## What it does

- Imports CSV, TSV, Excel, JSON/JSONL, and Parquet data.
- Profiles schema, missingness, duplicates, distributions, and correlations.
- Applies explainable cleaning operations with undo and transformation history.
- Converts natural-language requests into allow-listed cleaning plans with
  before/after preview and explicit approval.
- Builds filtered interactive Plotly dashboards.
- Runs comprehensive descriptive statistics, frequency/proportion tables, confidence
  intervals, Pearson/Spearman/Kendall correlation, covariance, and outlier analysis.
- Provides one-sample, independent and paired t-tests; Mann-Whitney U, Wilcoxon,
  ANOVA, Kruskal-Wallis, chi-square, effect sizes, and variance diagnostics.
- Includes Shapiro-Wilk, D'Agostino K² and Anderson-Darling normality checks,
  reproducible bootstrap intervals, and multiple OLS regression with coefficient
  inference and residual diagnostics.
- Produces an explainable ML-readiness score, data dictionary, potential leakage
  warnings, and sensitive-column checks.
- Creates reproducible train/validation/test splits.
- Compares classification or regression baselines, evaluates the winner on held-out
  data, explains feature importance, and exports the trained pipeline.
- Generates polished dataset documentation with local Qwen/Ollama, Gemini, Groq, or
  a deterministic professional fallback.
- Exports a complete ZIP containing original and cleaned data, optional ML splits and
  model, README, quality report, data dictionary, history, and a reproducibility
  script.

AI never executes generated code or edits the DataFrame directly. Natural-language
requests become a restricted JSON plan, are validated against known columns and safe
operations, and require user approval. Documentation writers receive schema and
aggregate quality metrics; raw records and example values are not sent to cloud
providers.

## Workspace

The responsive interface is organized into six focused areas:

1. **Understand** — preview and profile the dataset.
2. **Clean** — smart suggestions, natural-language plans, and history.
3. **Visualize** — charts and interactive dashboards.
4. **Analyze** — basic, inferential, robust and regression statistics plus
   privacy/leakage checks and ML readiness.
5. **AutoML** — reproducible splitting, model comparison, and downloads.
6. **Publish** — reports and the complete AI-assisted export package.

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

The dependency file includes the engines required for every advertised format,
including legacy `.xls`, `.xlsx`, Parquet, chart PNG, PDF, and statistical-model
exports. No separate browser installation is required for chart rendering.

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
