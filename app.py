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
st.caption(
    "Upload raw CSV files to generate fixture, bay, sample, "
    "heatmap, and abnormality analysis automatically."
)

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
st.sidebar.code(
    "Result, BayID, Date, StartTime, EndTime, air_pressure, Leakage_value",
    language="text"
)


# =========================
# Helper functions
# =========================

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


def fixture_sort_key(x):
    try:
        return float(str(x))
    except Exception:
        return 9999


def sample_sort_key(x):
    x = str(x)
    try:
        return int(x)
    except Exception:
        return x


def force_category_x(fig, category_order):
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=[str(x) for x in category_order]
    )
    return fig


def force_category_y(fig, category_order):
    fig.update_yaxes(
        type="category",
        categoryorder="array",
        categoryarray=[str(x) for x in category_order]
    )
    return fig


def clean_pivot_axis(pivot):
    pivot = pivot.copy()
    pivot.index = [str(x) for x in pivot.index]
    pivot.columns = [str(x) for x in pivot.columns]
    return pivot


def plot_heatmap(
    pivot,
    title,
    color_scale,
    text_format=".3f",
    zmin=None,
    zmax=None,
    colorbar_title=None
):
    """
    Use pivot.values + explicit x/y labels to avoid Plotly treating
    2.0 / 3.1 / 3.2 as continuous numeric axis.
    """
    pivot = clean_pivot_axis(pivot)

    x_labels = [str(x) for x in pivot.columns]
    y_labels = [str(y) for y in pivot.index]

    fig = px.imshow(
        pivot.values,
        x=x_labels,
        y=y_labels,
        text_auto=text_format,
        aspect="auto",
        color_continuous_scale=color_scale,
        zmin=zmin,
        zmax=zmax,
        title=title
    )

    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=x_labels
    )

    fig.update_yaxes(
        type="category",
        categoryorder="array",
        categoryarray=y_labels
    )

    if colorbar_title:
        fig.update_coloraxes(colorbar_title=colorbar_title)

    return fig


def parse_section_csv(uploaded_file, source_name, spec):
    rows, encoding = read_csv_rows(uploaded_file)

    if not rows:
        return pd.DataFrame(), encoding

    header = [str(x).strip() for x in rows[0]]

    try:
        leak_idx = [
            i for i, h in enumerate(header)
            if "Leakage_value" in h or "Leakage" in h
        ][0]

        pressure_idx = [
            i for i, h in enumerate(header)
            if "air_pressure" in h or "Pressure" in h
        ][0]

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

        # Detect fixture section, e.g. bay1 2.0 / bay2 3.1
        found_fixture = False

        for c in cells[:8]:
            if re.search(r"bay\s*\d+\s*\d\.\d", c, flags=re.I):
                current_fixture = re.search(r"(\d\.\d)", c).group(1)
                current_sample = None
                found_fixture = True
                break

        if found_fixture:
            continue

        # Detect sample number row
        if not cells[date_idx]:
            nums = [c for c in cells[:8] if re.fullmatch(r"\d+", c)]

            if nums and current_fixture:
                current_sample = nums[0]
                continue

        # Detect actual test data row
        if cells[date_idx] and cells[result_idx]:
            leakage = safe_float(cells[leak_idx])

            if np.isnan(leakage):
                continue

            pressure = safe_float(cells[pressure_idx])

            parsed.append({
                "Source": source_name,
                "Row": row_no,
                "Date": cells[date_idx],
                "Sample": str(current_sample).strip(),
                "Fixture": str(current_fixture).strip(),
                "Bay": str(cells[bay_idx]).strip(),
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

    df["Sample"] = df["Sample"].astype(str).str.strip()
    df["Fixture"] = df["Fixture"].astype(str).str.strip()
    df["Bay"] = df["Bay"].astype(str).str.strip()
    df["Result"] = df["Result"].astype(str).str.lower()

    df["Spec"] = spec
    df["Spec_Result"] = np.where(df["Leakage"] > spec, "NG", "OK")

    keep_cols = [
        "Source",
        "Sample",
        "Fixture",
        "Bay",
        "Result",
        "Pressure",
        "Leakage",
        "Spec",
        "Spec_Result"
    ]

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
            data.to_excel(
                writer,
                sheet_name=clean_name,
                index=True if data.index.name is not None else False
            )

    return output.getvalue()


# =========================
# Upload files
# =========================

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

        file_info.append({
            "File": f.name,
            "Rows parsed": len(parsed),
            "Encoding": encoding,
            "Format": fmt
        })

    except Exception as e:
        st.error(f"Failed to parse {f.name}: {e}")


if not parsed_list:
    st.stop()


# =========================
# Clean data
# =========================

df = pd.concat(parsed_list, ignore_index=True)
df = df.dropna(subset=["Leakage"])

df["Sample"] = df["Sample"].astype(str).str.strip()
df["Fixture"] = df["Fixture"].astype(str).str.strip()
df["Bay"] = df["Bay"].astype(str).str.strip()

# Clean abnormal string values
df["Sample"] = df["Sample"].replace(["None", "nan", ""], "Unknown")
df["Fixture"] = df["Fixture"].replace(["None", "nan", ""], "Unknown")
df["Bay"] = df["Bay"].replace(["None", "nan", ""], "Unknown")

df["NG_Flag"] = np.where(df["Leakage"] > SPEC, 1, 0)


fixture_order = sorted(
    [str(x) for x in df["Fixture"].dropna().unique()],
    key=fixture_sort_key
)

bay_order = sorted(
    [str(x) for x in df["Bay"].dropna().unique()],
    key=fixture_sort_key
)

sample_order = sorted(
    [str(x) for x in df["Sample"].dropna().unique()],
    key=sample_sort_key
)


# =========================
# File parsing result
# =========================

with st.expander("File parsing result", expanded=True):
    st.dataframe(pd.DataFrame(file_info), use_container_width=True)
    st.dataframe(df.head(50), use_container_width=True)


# =========================
# KPI
# =========================

total_n = len(df)
total_ng = int(df["NG_Flag"].sum())
ng_rate = total_ng / total_n if total_n else 0
max_leakage = df["Leakage"].max()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total rows", f"{total_n}")
c2.metric("NG count", f"{total_ng}")
c3.metric("NG rate", f"{ng_rate:.1%}")
c4.metric("Max leakage", f"{max_leakage:.4f}")


# =========================
# Summary
# =========================

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


# =========================
# Distribution charts
# =========================

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

    fig.add_hline(
        y=SPEC,
        line_dash="dash",
        annotation_text=f"Spec={SPEC}"
    )

    force_category_x(fig, fixture_order)

    st.plotly_chart(fig, use_container_width=True)


with col2:
    fig = px.strip(
        df,
        x="Fixture",
        y="Leakage",
        color="Bay",
        category_orders={
            "Fixture": fixture_order,
            "Bay": bay_order
        },
        title="Leakage Scatter by Fixture and Bay"
    )

    fig.add_hline(
        y=SPEC,
        line_dash="dash",
        annotation_text=f"Spec={SPEC}"
    )

    force_category_x(fig, fixture_order)

    st.plotly_chart(fig, use_container_width=True)


st.subheader("Histogram by fixture")

selected_fixture_for_hist = st.selectbox(
    "Select fixture for histogram",
    fixture_order
)

hist_df = df[df["Fixture"] == selected_fixture_for_hist]

fig = px.histogram(
    hist_df,
    x="Leakage",
    nbins=20,
    title=f"Leakage Histogram - Fixture {selected_fixture_for_hist}"
)

fig.add_vline(
    x=SPEC,
    line_dash="dash",
    annotation_text=f"Spec={SPEC}"
)

st.plotly_chart(fig, use_container_width=True)


# =========================
# Heatmaps
# =========================

st.header("3. Heatmaps")

pivot_mean_sample_fixture = df.pivot_table(
    index="Sample",
    columns="Fixture",
    values="Leakage",
    aggfunc="mean"
).reindex(index=sample_order, columns=fixture_order)

pivot_ng_sample_fixture = df.pivot_table(
    index="Sample",
    columns="Fixture",
    values="NG_Flag",
    aggfunc="sum"
).reindex(index=sample_order, columns=fixture_order)

pivot_mean_fixture_bay = df.pivot_table(
    index="Fixture",
    columns="Bay",
    values="Leakage",
    aggfunc="mean"
).reindex(index=fixture_order, columns=bay_order)


h1, h2 = st.columns(2)

with h1:
    fig = plot_heatmap(
        pivot=pivot_mean_sample_fixture,
        title="Mean Leakage by Sample and Fixture",
        color_scale="YlGnBu",
        text_format=".3f",
        zmin=0,
        zmax=SPEC,
        colorbar_title="Mean Leakage"
    )

    st.plotly_chart(fig, use_container_width=True)


with h2:
    fig = plot_heatmap(
        pivot=pivot_ng_sample_fixture,
        title="NG Count by Sample and Fixture",
        color_scale="OrRd",
        text_format=True,
        zmin=0,
        zmax=None,
        colorbar_title="NG Count"
    )

    st.plotly_chart(fig, use_container_width=True)


fig = plot_heatmap(
    pivot=pivot_mean_fixture_bay,
    title="Mean Leakage by Fixture and Bay",
    color_scale="YlGnBu",
    text_format=".3f",
    zmin=0,
    zmax=SPEC,
    colorbar_title="Mean Leakage"
)

st.plotly_chart(fig, use_container_width=True)


# =========================
# Abnormality analysis
# =========================

st.header("4. Abnormality analysis")

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

    selected_sample = st.selectbox(
        "Select abnormal sample for detail plot",
        abnormal_samples
    )

    temp = df[df["Sample"].astype(str) == selected_sample]

    fig = px.strip(
        temp,
        x="Fixture",
        y="Leakage",
        color="Bay",
        category_orders={
            "Fixture": fixture_order,
            "Bay": bay_order
        },
        title=f"Abnormal Sample Detail - Sample {selected_sample}"
    )

    fig.add_hline(
        y=SPEC,
        line_dash="dash",
        annotation_text=f"Spec={SPEC}"
    )

    force_category_x(fig, fixture_order)

    st.plotly_chart(fig, use_container_width=True)

else:
    st.success("No abnormal groups found by current criteria.")

# =========================
# Before vs After comparison
# =========================

st.header("4. Before vs After Comparison")

st.caption(
    "Upload previous / before data here. The app will compare it with the current uploaded data above."
)


def parse_previous_dataset(uploaded_file, spec):
    """
    Parse previous 80A dataset.
    This function supports:
    1. Real CSV files
    2. Excel files renamed as .csv
    3. Section-style data with rows like 'bay1 2.0', 'bay1 3.1', 'bay1 3.2'
    """

    raw = uploaded_file.getvalue()

    # Case 1: Excel file renamed as .csv
    # Excel xlsx files usually start with PK
    if raw[:2] == b"PK":
        excel_file = pd.ExcelFile(io.BytesIO(raw))
        sheet_name = excel_file.sheet_names[0]
        raw_df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name)

    else:
        # Case 2: normal CSV
        encodings = ["gbk", "utf-8-sig", "utf-8", "latin1"]
        raw_df = None

        for enc in encodings:
            try:
                raw_df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                break
            except Exception:
                pass

        if raw_df is None:
            raise ValueError("Cannot read previous dataset. Please check file format.")

    # Find leakage column
    leakage_cols = [
        c for c in raw_df.columns
        if "Leakage_value" in str(c) or "Leakage" in str(c) or "leakage" in str(c).lower()
    ]

    if not leakage_cols:
        raise ValueError("Cannot find leakage column in previous dataset.")

    leakage_col = leakage_cols[0]

    # Find pressure column if available
    pressure_cols = [
        c for c in raw_df.columns
        if "air_pressure" in str(c) or "Pressure" in str(c)
    ]
    pressure_col = pressure_cols[0] if pressure_cols else None

    # Find basic columns
    sn_col = "SN" if "SN" in raw_df.columns else None
    result_col = "Result" if "Result" in raw_df.columns else None
    bay_col = "BayID" if "BayID" in raw_df.columns else None

    parsed = []
    current_fixture = None
    current_bay_from_section = None

    for idx, row in raw_df.iterrows():

        # Detect section marker like "bay1 2.0"
        row_values = [str(v).strip() for v in row.values if pd.notna(v)]
        row_joined = " ".join(row_values)

        section_match = re.search(r"bay\s*(\d+)\s*(\d\.\d)", row_joined, flags=re.I)

        if section_match:
            current_bay_from_section = section_match.group(1)
            current_fixture = section_match.group(2)
            continue

        leakage = pd.to_numeric(row.get(leakage_col), errors="coerce")

        if pd.isna(leakage):
            continue

        result = str(row.get(result_col, "")).strip().lower() if result_col else ""
        bay = row.get(bay_col, current_bay_from_section) if bay_col else current_bay_from_section

        if pd.isna(bay):
            bay = current_bay_from_section

        pressure = (
            pd.to_numeric(row.get(pressure_col), errors="coerce")
            if pressure_col
            else np.nan
        )

        sn = str(row.get(sn_col, "Unknown")).strip() if sn_col else "Unknown"

        parsed.append({
            "Source": uploaded_file.name,
            "Dataset": "Before",
            "Row": idx,
            "Sample": sn if sn not in ["nan", "None", ""] else "Unknown",
            "Fixture": str(current_fixture).strip() if current_fixture else "Unknown",
            "Bay": str(int(bay)) if pd.notna(bay) and str(bay).replace(".0", "").isdigit() else str(bay),
            "Result": result,
            "Pressure": pressure,
            "Leakage": float(leakage),
            "Spec": spec,
            "Spec_Result": "NG" if float(leakage) > spec else "OK",
            "NG_Flag": 1 if float(leakage) > spec else 0,
        })

    previous_df = pd.DataFrame(parsed)

    if previous_df.empty:
        raise ValueError("No valid leakage rows parsed from previous dataset.")

    previous_df["Sample"] = previous_df["Sample"].astype(str).str.strip()
    previous_df["Fixture"] = previous_df["Fixture"].astype(str).str.strip()
    previous_df["Bay"] = previous_df["Bay"].astype(str).str.strip()
    previous_df["Leakage"] = pd.to_numeric(previous_df["Leakage"], errors="coerce")
    previous_df["NG_Flag"] = np.where(previous_df["Leakage"] > spec, 1, 0)
    previous_df["Spec_Result"] = np.where(previous_df["Leakage"] > spec, "NG", "OK")

    return previous_df


def compare_summary_table(data, group_cols):
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
            NG_Count=("NG_Flag", "sum"),
        )
        .reset_index()
    )

    summary["IQR"] = summary["Q3"] - summary["Q1"]
    summary["Range"] = summary["Max"] - summary["Min"]
    summary["NG_Rate"] = summary["NG_Count"] / summary["N"]

    return summary


previous_file = st.file_uploader(
    "Upload previous / before dataset",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=False,
    key="previous_dataset_uploader"
)

if previous_file is not None:

    try:
        before_df = parse_previous_dataset(previous_file, SPEC)

        after_df = df.copy()
        after_df["Dataset"] = "After"

        # Keep same columns
        keep_cols = [
            "Dataset",
            "Source",
            "Sample",
            "Fixture",
            "Bay",
            "Result",
            "Pressure",
            "Leakage",
            "Spec",
            "Spec_Result",
            "NG_Flag"
        ]

        for col in keep_cols:
            if col not in after_df.columns:
                after_df[col] = np.nan

        after_df = after_df[keep_cols]
        before_df = before_df[keep_cols]

        compare_df = pd.concat([before_df, after_df], ignore_index=True)

        compare_df["Fixture"] = compare_df["Fixture"].astype(str).str.strip()
        compare_df["Bay"] = compare_df["Bay"].astype(str).str.strip()
        compare_df["Sample"] = compare_df["Sample"].astype(str).str.strip()

        compare_fixture_order = sorted(
            [str(x) for x in compare_df["Fixture"].dropna().unique()],
            key=fixture_sort_key
        )

        compare_bay_order = sorted(
            [str(x) for x in compare_df["Bay"].dropna().unique()],
            key=fixture_sort_key
        )

        st.subheader("4.1 Parsed previous data")
        st.dataframe(before_df.head(50), use_container_width=True)

        # =========================
        # Summary comparison
        # =========================

        st.subheader("4.2 Summary comparison by fixture")

        compare_summary_fixture = compare_summary_table(
            compare_df,
            ["Dataset", "Fixture"]
        )

        st.dataframe(compare_summary_fixture, use_container_width=True)

        # Pivot summary for easier reading
        summary_pivot = compare_summary_fixture.pivot(
            index="Fixture",
            columns="Dataset",
            values=["Mean", "Median", "Std", "Max", "NG_Rate"]
        )

        st.subheader("4.3 Pivot summary")
        st.dataframe(summary_pivot, use_container_width=True)

        # =========================
        # Mean leakage comparison bar chart
        # =========================

        st.subheader("4.4 Mean leakage comparison")

        fig = px.bar(
            compare_summary_fixture,
            x="Fixture",
            y="Mean",
            color="Dataset",
            barmode="group",
            category_orders={
                "Fixture": compare_fixture_order,
                "Dataset": ["Before", "After"]
            },
            title="Before vs After - Mean Leakage by Fixture",
            text_auto=".3f"
        )

        fig.add_hline(
            y=SPEC,
            line_dash="dash",
            annotation_text=f"Spec={SPEC}"
        )

        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=compare_fixture_order
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # NG rate comparison bar chart
        # =========================

        st.subheader("4.5 NG rate comparison")

        compare_summary_fixture["NG_Rate_Percent"] = (
            compare_summary_fixture["NG_Rate"] * 100
        )

        fig = px.bar(
            compare_summary_fixture,
            x="Fixture",
            y="NG_Rate_Percent",
            color="Dataset",
            barmode="group",
            category_orders={
                "Fixture": compare_fixture_order,
                "Dataset": ["Before", "After"]
            },
            title="Before vs After - NG Rate by Fixture",
            text_auto=".1f"
        )

        fig.update_yaxes(title="NG Rate (%)")

        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=compare_fixture_order
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # Boxplot comparison
        # =========================

        st.subheader("4.6 Leakage distribution comparison")

        fig = px.box(
            compare_df,
            x="Fixture",
            y="Leakage",
            color="Dataset",
            points="all",
            category_orders={
                "Fixture": compare_fixture_order,
                "Dataset": ["Before", "After"]
            },
            title="Before vs After - Leakage Distribution by Fixture"
        )

        fig.add_hline(
            y=SPEC,
            line_dash="dash",
            annotation_text=f"Spec={SPEC}"
        )

        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=compare_fixture_order
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # Fixture + Bay comparison
        # =========================

        st.subheader("4.7 Fixture / Bay comparison")

        compare_summary_bay_fixture = compare_summary_table(
            compare_df,
            ["Dataset", "Bay", "Fixture"]
        )

        st.dataframe(compare_summary_bay_fixture, use_container_width=True)

        fig = px.bar(
            compare_summary_bay_fixture,
            x="Fixture",
            y="Mean",
            color="Dataset",
            facet_col="Bay",
            barmode="group",
            category_orders={
                "Fixture": compare_fixture_order,
                "Bay": compare_bay_order,
                "Dataset": ["Before", "After"]
            },
            title="Before vs After - Mean Leakage by Fixture and Bay",
            text_auto=".3f"
        )

        fig.add_hline(
            y=SPEC,
            line_dash="dash",
            annotation_text=f"Spec={SPEC}"
        )

        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=compare_fixture_order
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # Improvement table
        # =========================

        st.subheader("4.8 Improvement summary")

        before_summary = compare_summary_fixture[
            compare_summary_fixture["Dataset"] == "Before"
        ][["Fixture", "Mean", "NG_Rate"]].rename(
            columns={
                "Mean": "Before_Mean",
                "NG_Rate": "Before_NG_Rate"
            }
        )

        after_summary = compare_summary_fixture[
            compare_summary_fixture["Dataset"] == "After"
        ][["Fixture", "Mean", "NG_Rate"]].rename(
            columns={
                "Mean": "After_Mean",
                "NG_Rate": "After_NG_Rate"
            }
        )

        improvement = before_summary.merge(
            after_summary,
            on="Fixture",
            how="outer"
        )

        improvement["Mean_Reduction"] = (
            improvement["Before_Mean"] - improvement["After_Mean"]
        )

        improvement["Mean_Reduction_%"] = (
            improvement["Mean_Reduction"] / improvement["Before_Mean"]
        )

        improvement["NG_Rate_Reduction"] = (
            improvement["Before_NG_Rate"] - improvement["After_NG_Rate"]
        )

        improvement = improvement.sort_values(
            by="Fixture",
            key=lambda x: x.map(fixture_sort_key)
        )

        st.dataframe(improvement, use_container_width=True)

        # =========================
        # Download comparison Excel
        # =========================

        comparison_excel_bytes = to_excel_download({
            "Before_Raw_Parsed": before_df,
            "After_Raw_Parsed": after_df,
            "Compare_All_Data": compare_df,
            "Compare_By_Fixture": compare_summary_fixture,
            "Compare_By_Bay_Fixture": compare_summary_bay_fixture,
            "Improvement_Summary": improvement,
        })

        st.download_button(
            label="Download before vs after comparison Excel",
            data=comparison_excel_bytes,
            file_name="LIM_Seal_Before_After_Comparison.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Failed to run before vs after comparison: {e}")
# =========================
# Download report
# =========================

st.header("5. Download report data")

excel_bytes = to_excel_download({
    "Raw_Parsed": df,
    "Summary_By_Fixture": summary_fixture,
    "Summary_By_Bay_Fixture": summary_bay_fixture,
    "Summary_By_Sample_Fixture": summary_sample_fixture,
    "Summary_Sample_Bay_Fixture": summary_sample_bay_fixture,
    "Mean_Sample_Fixture": pivot_mean_sample_fixture,
    "NG_Sample_Fixture": pivot_ng_sample_fixture,
    "Mean_Fixture_Bay": pivot_mean_fixture_bay,
    "Abnormal_Groups": abnormal if not abnormal.empty else pd.DataFrame(),
})

st.download_button(
    label="Download Excel analysis report",
    data=excel_bytes,
    file_name="LIM_Seal_DOE_Analysis_Report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
