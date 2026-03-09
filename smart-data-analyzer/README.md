# Smart Data Analyzer

**Smart Data Analyzer** is a Streamlit web application designed as a simpler but more powerful alternative to Excel. It enables beginners, students, researchers, and analysts to upload CSV/Excel datasets, clean and preprocess data, generate interactive visualizations, perform statistical analysis, and export results—without writing code.

---

## ✅ Key Features

- **Upload CSV & Excel** files
- **Data preview** with schema, missing values, and quick insights
- **Cleaning tools**: remove missing values, fill missing, remove duplicates, drop/rename columns, type conversion, filtering, sorting
- **Interactive visualizations**: histograms, bar charts, line charts, scatter plots, box plots, correlation heatmaps, distribution plots
- **Statistical analysis**: descriptive stats, correlation/covariance matrices, linear regression, outlier detection (Z-score & IQR), hypothesis testing
- **Export** cleaned datasets, charts (PNG), and summaries
- **Smart insights**: auto-detect numeric columns, data quality indicators, suggested charts

---

## 🧰 Technologies Used

- Python
- Streamlit
- Pandas / NumPy
- Plotly
- SciPy
- scikit-learn
- Matplotlib / Seaborn
- Openpyxl

---

## 🚀 Getting Started (Local)

### 1) Clone the repository

```bash
git clone https://github.com/<your-username>/Smart-Data-Analyzer.git
cd Smart-Data-Analyzer/smart-data-analyzer
```

### 2) Create and activate a virtual environment

On macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

On Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run the app

```bash
streamlit run app.py
```

Then open the URL shown in your browser (usually http://localhost:8501).

---

## 📦 Project Structure

```
smart-data-analyzer/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── modules/
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── visualization.py
│   ├── statistics_tools.py
│   └── export_tools.py
├── assets/
│   └── logo.png
└── sample_data/
    └── example_dataset.csv
```

---

## ☁️ Deploy to Streamlit Community Cloud

1. Create a GitHub repository and push this project.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app** → connect your GitHub account.
4. Select your repository and the correct branch (`main`).
5. Set the **Main file path** to `app.py`.
6. Click **Deploy**.

---

## 🧩 Tips

- Use the **Data Preview** tab to understand your data structure quickly.
- Use the **Cleaning Tools** sidebar to iteratively clean the dataset.
- Try different chart types in **Visualization** to surface trends.
- Export cleaned datasets when you're ready to share results.

---

## 🛠️ Customization

You can extend the app by adding more cleaning or analysis features in `modules/` and expose them via the Streamlit UI.

---

Enjoy exploring your data! 🎉
