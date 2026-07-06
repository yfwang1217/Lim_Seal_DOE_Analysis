
# LIM Seal Self-Seal DOE Analysis App

This is a Streamlit web app for LIM seal self-seal DOE analysis.

## Features

- Upload one or multiple CSV files
- Auto-parse Everwin-style raw CSV files
- Generate summary tables, boxplots, scatter plots, histograms, heatmaps, correlation analysis, and abnormal sample detail
- Download Excel analysis report

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repo.
2. Upload `app.py` and `requirements.txt`.
3. Go to Streamlit Community Cloud.
4. Connect GitHub.
5. Choose your repo, branch, and `app.py`.
6. Deploy.
7. Share the generated app URL with customers.

## Notes

- Leakage spec default is 0.05.
- NG if leakage > spec.
- The app supports multiple CSV files uploaded at the same time.
- Uploaded files are processed in the browser session. For strict data security, deploy internally.
