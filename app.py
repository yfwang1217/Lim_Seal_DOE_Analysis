
import io
import re
import csv
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="LIM Seal Self-Seal DOE Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("LIM Seal Self-Seal DOE Analysis")
st.caption("Upload Everwin / ICT raw CSV files to generate fixture, bay, sample, heatmap, and abnormality analysis automatically.")

st.sidebar.header("Settings")
SPEC = st.sidebar.number_input(
    "Leakage spec, NG if leakage > spec",
    min_value=0.0,
    value=0.05,
    step=0.001,
    format="%.3f"
)

st.sidebar.markdown("---")
st.sidebar.write("Expected columns or raw section format:")
st.sidebar.code("Result, BayID, Date, StartTime, EndTime, air_pressure, Leakage_value", language="text")


def read_csv_rows(uploaded_file):
    raw = uploaded_file.getvalue()
    encodings = ["gbk", "utf-8-sig", "utf-8", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            text = raw.decode(enc)
            rows = list(csv.reader(io.StringIO(text)))
            return rows, enc
        except Exception as e:
            last_error = e

    raise ValueError(f"Cannot read {uploaded_file.name}. Last error: {last_error}")


def safe_float(x):
    try:
        return float(str(x).strip())
    except Exception:
        return np.nan


def parse_section_csv(uploaded_file, source_name, spec):
    rows, encoding = read_csv_rows(uploaded_file)
    if not rows:
        return pd.DataFrame(), encoding

    header = [str(x).strip() for x in rows[0]]

    try:
        leak_idx = [i for i, h in enumerate(header) if "Leakage_value" in h or "Leakage" in h][0]
        pressure_idx = [i for i, h in enumerate(header) if "air_pressure" in h or "Pressure" in h][0]
        result_idx = header.index("Result")
        bay_idx = header.index("BayID")
        date_idx = header.index("Date")
        start_idx = header.index("StartTime")
        end_idx = header.index("EndTime")
    except Exception:
        return None, encoding

    current_fixture = None
    current_sample = None
    parsed = []

    for row_no, r in enumerate(rows[1:], start=1):
        r = r + [""] * (len(header) - len(r))
        cells = [str(c).strip() for c in r]

        found_fixture = False
        for c in cells[:8]:
            if re.search(r"bay\s*\d+\s*\d\.\d", c, flags=re.I):
                current_fixture = re.search(r"(\d\.\d)", c).group(1)
                current_sample = None
                found_fixture = True
                break
        if found_fixture:
            continue

        if not cells[date_idx]:
            nums = [c for c in cells[:8] if re.fullmatch(r"\d+", c)]
            if nums and current_fixture:
                current_sample = nums[0]
                continue

        if cells[date_idx] and cells[result_idx]:
            leakage = safe_float(cells[leak_idx])
            if np.isnan(leakage):
                continue

            pressure = safe_float(cells[pressure_idx])

            parsed.append({
                "Source": source_name,
                "Row": row_no,
                "Date": cells[date_idx],
                "Sample": str(current_sample),
                "Fixture": str(current_fixture),
                "Bay": str(cells[bay_idx]),
                "Result": str(cells[result_idx]).strip().lower(),
                "StartTime": cells[start_idx],
                "EndTime": cells[end_idx],
                "Pressure": pressure,
                "Leakage": leakage,
                "Spec": spec,
                "Spec_Result": "NG" if leakage > spec else "OK",
            })

    return pd.DataFrame(parsed), encoding


def parse_flat_csv(uploaded_file, source_name, spec):
    raw = uploaded_file.getvalue()
    encodings = ["gbk", "utf-8-sig", "utf-8", "latin1"]
    df = None
    encoding_used = None

    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            encoding_used = enc
            break
        except Exception:
            pass

    if df is None:
        raise ValueError(f"Cannot read {uploaded_file.name}")

    col_map = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ["sample", "sn"]:
            col_map[c] = "Sample"
        elif lc in ["fixture", "carrier", "carrier_id", "carrier version"]:
            col_map[c] = "Fixture"
        elif lc in ["bay", "bayid", "bay id"]:
            col_map[c] = "Bay"
        elif "leakage" in lc or "leak" in lc:
            col_map[c] = "Leakage"
        elif lc == "result":
            col_map[c] = "Result"
        elif "pressure" in lc:
            col_map[c] = "Pressure"

    df = df.rename(columns=col_map)

    if "Leakage" not in df.columns:
        raise ValueError("Cannot find Leakage column. Please check file format.")

    if "Sample" not in df.columns:
        df["Sample"] = np.arange(1, len(df) + 1).astype(str)
    if "Fixture" not in df.columns:
        df["Fixture"] = "Unknown"
    if "Bay" not in df.columns:
        df["Bay"] = "Unknown"
    if "Result" not in df.columns:
        df["Result"] = ""

    df["Source"] = source_name
    df["Leakage"] = pd.to_numeric(df["Leakage"], errors="coerce")
    df["Pressure"] = pd.to_numeric(df.get("Pressure", np.nan), errors="coerce")
    df["Sample"] = df["Sample"].astype(str)
    df["Fixture"] = df["Fixture"].astype(str)
    df["Bay"] = df["Bay"].astype(str)
    df["Result"] = df["Result"].astype(str).str.lower()
    df["Spec"] = spec
    df["Spec_Result"] = np.where(df["Leakage"] > spec, "NG", "OK")

    keep_cols = ["Source", "Sample", "Fixture", "Bay", "Result", "Pressure", "Leakage", "Spec", "Spec_Result"]
    return df[keep_cols], encoding_used


def parse_uploaded_file(uploaded_file, spec):
    source_name = uploaded_file.name
    section_df, enc = parse_section_csv(uploaded_file, source_name, spec)

    if section_df is not None and len(section_df) > 0:
        return section_df, enc, "section format"

    flat_df, enc2 = parse_flat_csv(uploaded_file, source_name, spec)
    return flat_df, enc2, "flat format"


def summary_table(data, group_cols):
    if data.empty:
        return pd.DataFrame()

    summary = (
        data
        .groupby(group_cols, dropna=False)
        .agg(
            N=("Leakage", "count"),
            Mean=("Leakage", "mean"),
            Median=("Leakage", "median"),
            Std=("Leakage", "std"),
            Min=("Leakage", "min"),
            Q1=("Leakage", lambda x: x.quantile(0.25)),
            Q3=("Leakage", lambda x: x.quantile(0.75)),
            Max=("Leakage", "max"),
            NG_Count=("Spec_Result", lambda x: (x == "NG").sum()),
            OK_Count=("Spec_Result", lambda x: (x == "OK").sum()),
        )
        .reset_index()
    )

    summary["IQR"] = summary["Q3"] - summary["Q1"]
    summary["Range"] = summary["Max"] - summary["Min"]
    summary["NG_Rate"] = summary["NG_Count"] / summary["N"]
    return summary


def to_excel_download(dataframes):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, data in dataframes.items():
            clean_name = name[:31]
            data.to_excel(writer, sheet_name=clean_name, index=True if data.index.name is not None else False)
    return output.getvalue()


uploaded_files = st.file_uploader(
    "Upload one or multiple CSV files",
    type=["csv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload CSV files to start analysis. You can upload Bay1 and Bay2 files together.")
    st.stop()

parsed_list = []
file_info = []

for f in uploaded_files:
    try:
        parsed, encoding, fmt = parse_uploaded_file(f, SPEC)
        parsed_list.append(parsed)
        file_info.append({"File": f.name, "Rows parsed": len(parsed), "Encoding": encoding, "Format": fmt})
    except Exception as e:
        st.error(f"Failed to parse {f.name}: {e}")

if not parsed_list:
    st.stop()

df = pd.concat(parsed_list, ignore_index=True)
df = df.dropna(subset=["Leakage"])
df["Sample"] = df["Sample"].astype(str)
df["Fixture"] = df["Fixture"].astype(str)
df["Bay"] = df["Bay"].astype(str)
df["NG_Flag"] = np.where(df["Leakage"] > SPEC, 1, 0)

def fixture_sort_key(x):
    try:
        return float(x)
    except Exception:
        return 9999

fixture_order = sorted(df["Fixture"].dropna().unique(), key=fixture_sort_key)
bay_order = sorted(df["Bay"].dropna().unique())

with st.expander("File parsing result", expanded=True):
    st.dataframe(pd.DataFrame(file_info), use_container_width=True)
    st.dataframe(df.head(50), use_container_width=True)

total_n = len(df)
total_ng = int(df["NG_Flag"].sum())
ng_rate = total_ng / total_n if total_n else 0
max_leakage = df["Leakage"].max()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total rows", f"{total_n}")
c2.metric("NG count", f"{total_ng}")
c3.metric("NG rate", f"{ng_rate:.1%}")
c4.metric("Max leakage", f"{max_leakage:.4f}")

summary_fixture = summary_table(df, ["Fixture"])
summary_bay_fixture = summary_table(df, ["Bay", "Fixture"])
summary_sample_fixture = summary_table(df, ["Sample", "Fixture"])
summary_sample_bay_fixture = summary_table(df, ["Sample", "Bay", "Fixture"])

st.header("1. Summary tables")

tab1, tab2, tab3, tab4 = st.tabs([
    "By Fixture",
    "By Bay + Fixture",
    "By Sample + Fixture",
    "By Sample + Bay + Fixture"
])

with tab1:
    st.dataframe(summary_fixture, use_container_width=True)
with tab2:
    st.dataframe(summary_bay_fixture, use_container_width=True)
with tab3:
    st.dataframe(summary_sample_fixture, use_container_width=True)
with tab4:
    st.dataframe(summary_sample_bay_fixture, use_container_width=True)

st.header("2. Distribution charts")

col1, col2 = st.columns(2)

with col1:
    fig = px.box(
        df,
        x="Fixture",
        y="Leakage",
        points="all",
        category_orders={"Fixture": fixture_order},
        title="Leakage Distribution by Fixture"
    )
    fig.add_hline(y=SPEC, line_dash="dash", annotation_text=f"Spec={SPEC}")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.strip(
        df,
        x="Fixture",
        y="Leakage",
        color="Bay",
        category_orders={"Fixture": fixture_order},
        title="Leakage Scatter by Fixture and Bay"
    )
    fig.add_hline(y=SPEC, line_dash="dash", annotation_text=f"Spec={SPEC}")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Histogram by fixture")
selected_fixture_for_hist = st.selectbox("Select fixture for histogram", fixture_order)
hist_df = df[df["Fixture"] == selected_fixture_for_hist]
fig = px.histogram(
    hist_df,
    x="Leakage",
    nbins=20,
    title=f"Leakage Histogram - Fixture {selected_fixture_for_hist}"
)
fig.add_vline(x=SPEC, line_dash="dash", annotation_text=f"Spec={SPEC}")
st.plotly_chart(fig, use_container_width=True)

st.header("3. Heatmaps")

pivot_mean_sample_fixture = df.pivot_table(
    index="Sample", columns="Fixture", values="Leakage", aggfunc="mean"
).reindex(columns=fixture_order)

pivot_ng_sample_fixture = df.pivot_table(
    index="Sample", columns="Fixture", values="NG_Flag", aggfunc="sum"
).reindex(columns=fixture_order)

pivot_mean_fixture_bay = df.pivot_table(
    index="Fixture", columns="Bay", values="Leakage", aggfunc="mean"
).reindex(index=fixture_order, columns=bay_order)

pivot_max_sample_fixture = df.pivot_table(
    index="Sample", columns="Fixture", values="Leakage", aggfunc="max"
).reindex(columns=fixture_order)

h1, h2 = st.columns(2)

with h1:
    fig = px.imshow(
        pivot_mean_sample_fixture,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale="YlGnBu",
        zmin=0,
        zmax=SPEC,
        title="Mean Leakage by Sample and Fixture"
    )
    st.plotly_chart(fig, use_container_width=True)

with h2:
    fig = px.imshow(
        pivot_ng_sample_fixture,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="OrRd",
        title="NG Count by Sample and Fixture"
    )
    st.plotly_chart(fig, use_container_width=True)

h3, h4 = st.columns(2)

with h3:
    fig = px.imshow(
        pivot_mean_fixture_bay,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale="YlGnBu",
        zmin=0,
        zmax=SPEC,
        title="Mean Leakage by Fixture and Bay"
    )
    st.plotly_chart(fig, use_container_width=True)

with h4:
    fig = px.imshow(
        pivot_max_sample_fixture,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale="YlOrRd",
        zmin=0,
        zmax=SPEC,
        title="Max Leakage by Sample and Fixture"
    )
    st.plotly_chart(fig, use_container_width=True)

st.header("4. Correlation analysis")

sample_fixture_mean = (
    df.groupby(["Sample", "Fixture"])["Leakage"]
    .mean()
    .reset_index()
)

wide_sample_fixture = sample_fixture_mean.pivot(
    index="Sample", columns="Fixture", values="Leakage"
).reindex(columns=fixture_order)

corr_method = st.radio("Correlation method", ["pearson", "spearman"], horizontal=True)
corr = wide_sample_fixture.corr(method=corr_method)

fig = px.imshow(
    corr,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title=f"Correlation Heatmap - {corr_method.title()}"
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Wide table used for correlation"):
    st.dataframe(wide_sample_fixture, use_container_width=True)

st.header("5. Abnormality analysis")

abnormal = summary_sample_bay_fixture[
    (summary_sample_bay_fixture["Mean"] > SPEC) |
    (summary_sample_bay_fixture["Max"] > SPEC) |
    (summary_sample_bay_fixture["NG_Count"] > 0)
].copy()

if not abnormal.empty:
    abnormal = abnormal.sort_values(
        by=["NG_Count", "Max", "Mean"],
        ascending=[False, False, False]
    )
    st.dataframe(abnormal, use_container_width=True)

    abnormal_samples = abnormal["Sample"].astype(str).unique().tolist()
    selected_sample = st.selectbox("Select abnormal sample for detail plot", abnormal_samples)

    temp = df[df["Sample"].astype(str) == selected_sample]

    fig = px.strip(
        temp,
        x="Fixture",
        y="Leakage",
        color="Bay",
        category_orders={"Fixture": fixture_order},
        title=f"Abnormal Sample Detail - Sample {selected_sample}"
    )
    fig.add_hline(y=SPEC, line_dash="dash", annotation_text=f"Spec={SPEC}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.success("No abnormal groups found by current criteria.")

st.header("6. Download report data")

excel_bytes = to_excel_download({
    "Raw_Parsed": df,
    "Summary_By_Fixture": summary_fixture,
    "Summary_By_Bay_Fixture": summary_bay_fixture,
    "Summary_By_Sample_Fixture": summary_sample_fixture,
    "Summary_Sample_Bay_Fixture": summary_sample_bay_fixture,
    "Mean_Sample_Fixture": pivot_mean_sample_fixture,
    "NG_Sample_Fixture": pivot_ng_sample_fixture,
    "Mean_Fixture_Bay": pivot_mean_fixture_bay,
    "Max_Sample_Fixture": pivot_max_sample_fixture,
    "Correlation": corr,
    "Abnormal_Groups": abnormal if not abnormal.empty else pd.DataFrame(),
})

st.download_button(
    label="Download Excel analysis report",
    data=excel_bytes,
    file_name="LIM_Seal_DOE_Analysis_Report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
