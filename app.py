
import os
import html
import uuid
from datetime import datetime

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import (
    Flask,
    request,
    send_from_directory
)

from werkzeug.utils import secure_filename

from etl_engine import (
    run_data_quality_analysis,
    build_cleaning_preview
)

from anomaly_detection import (
    prepare_anomaly_data,
    run_all_anomaly_models,
    export_anomaly_results
)

from sql_queries import (
    export_sql_analysis
)


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
EXPORT_FOLDER = os.path.join(BASE_DIR, "exports")

VISUALIZATION_FOLDER = os.path.join(
    EXPORT_FOLDER,
    "visualizations"
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "flowforge.db"
)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "json"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPORT_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)

AUDIT_LOG = os.path.join(EXPORT_FOLDER, "audit_log.csv")


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def read_dataset(filepath, extension):

    if extension == "csv":
        return pd.read_csv(filepath)

    elif extension == "xlsx":
        return pd.read_excel(filepath)

    elif extension == "json":
        return pd.read_json(filepath)

    raise ValueError("Unsupported file format.")


def make_table(dataframe, limit=10):

    if dataframe is None or dataframe.empty:
        return "<p>No records found.</p>"

    return dataframe.head(limit).to_html(
        index=False,
        border=0,
        classes="data-table",
        na_rep="Missing",
        escape=True
    )


def validate_cleaning_actions(duplicate_action, missing_action):

    valid_actions = {"keep", "remove", "quarantine"}

    if duplicate_action not in valid_actions:
        raise ValueError("Invalid duplicate handling action.")

    if missing_action not in valid_actions:
        raise ValueError("Invalid missing-value handling action.")


def read_uploaded_file(filename):

    filename = secure_filename(filename)

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.isfile(filepath):
        raise FileNotFoundError("Uploaded dataset not found.")

    extension = filename.rsplit(".", 1)[1].lower()

    return read_dataset(filepath, extension)


def format_number(value):

    try:
        return f"{float(value):,.0f}"
    except (ValueError, TypeError):
        return "0"


def format_currency(value):

    try:
        return f"{float(value):,.2f}"
    except (ValueError, TypeError):
        return "0.00"


def page_template(title, content):

    return f"""
    <!DOCTYPE html>
    <html>
    <head>

        <title>{html.escape(title)}</title>

        <meta name="viewport" content="width=device-width, initial-scale=1">

        <style>

            * {{
                box-sizing: border-box;
            }}

            body {{
                font-family: Arial, sans-serif;
                background: #f4f6f9;
                margin: 0;
                padding: 25px;
                color: #263238;
            }}

            .container {{
                max-width: 1250px;
                margin: auto;
            }}

            h1, h2 {{
                color: #17365d;
            }}

            h2 {{
                margin-top: 30px;
                border-bottom: 2px solid #dce3eb;
                padding-bottom: 8px;
            }}

            .subtitle {{
                color: #64748b;
            }}

            .section {{
                background: white;
                padding: 22px;
                margin-top: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 8px #00000010;
                overflow-x: auto;
            }}

            .kpi-grid {{
                display: grid;
                grid-template-columns:
                    repeat(auto-fit, minmax(170px, 1fr));
                gap: 15px;
                margin-top: 25px;
            }}

            .kpi {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 8px #00000010;
            }}

            .kpi h3 {{
                font-size: 14px;
                color: #64748b;
            }}

            .kpi p {{
                font-size: 25px;
                font-weight: bold;
                color: #17365d;
                overflow-wrap: anywhere;
            }}

            .data-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
                font-size: 13px;
            }}

            .data-table th {{
                background: #17365d;
                color: white;
                padding: 10px;
                text-align: left;
            }}

            .data-table td {{
                padding: 9px;
                border-bottom: 1px solid #e5e7eb;
            }}

            .rule-grid {{
                display: grid;
                grid-template-columns:
                    repeat(auto-fit, minmax(280px, 1fr));
                gap: 15px;
            }}

            .rule-card {{
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 15px;
                overflow-x: auto;
            }}

            .notice {{
                background: #eaf3ff;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
            }}

            .form-box {{
                background: #f8fafc;
                padding: 20px;
                border-radius: 8px;
                margin-top: 15px;
            }}

            label {{
                display: block;
                margin-top: 15px;
                font-weight: bold;
            }}

            select, input[type=file] {{
                padding: 10px;
                width: 100%;
                max-width: 400px;
                margin-top: 8px;
                border: 1px solid #cbd5e1;
                border-radius: 5px;
                background: white;
            }}

            button, .button {{
                display: inline-block;
                background: #17365d;
                color: white;
                padding: 12px 20px;
                border: none;
                border-radius: 6px;
                text-decoration: none;
                margin-top: 18px;
                cursor: pointer;
            }}

            button:hover, .button:hover {{
                background: #24548a;
            }}

            .success {{
                background: #e8f5e9;
                padding: 15px;
                border-radius: 8px;
            }}

            .warning {{
                background: #fff4e5;
                padding: 15px;
                border-radius: 8px;
            }}

            .chart {{
                width: 100%;
                max-width: 100%;
                height: auto;
                margin-top: 15px;
                border-radius: 8px;
            }}

            .chart-grid {{
                display: grid;
                grid-template-columns:
                    repeat(auto-fit, minmax(350px, 1fr));
                gap: 20px;
            }}

            @media(max-width: 600px) {{

                body {{
                    padding: 12px;
                }}

                .chart-grid {{
                    grid-template-columns: 1fr;
                }}

            }}

        </style>

    </head>

    <body>

        <div class="container">
            {content}
        </div>

    </body>
    </html>
    """


def error_page(message):

    content = f"""
        <h1>FlowForge</h1>

        <div class="warning">
            <h2>Something went wrong</h2>
            <p>{html.escape(str(message))}</p>
        </div>

        <a class="button" href="/upload">Try Again</a>
        <a class="button" href="/">Home</a>
    """

    return page_template("FlowForge Error", content), 400


# ============================================================
# 3. DATA QUALITY DASHBOARD
# ============================================================

def dashboard_page(filename, results, cleaning_preview=None):

    missing_table = make_table(results["null_report"])
    type_table = make_table(results["data_types"])
    duplicate_table = make_table(results["duplicate_records"])
    original_preview = make_table(results["dataset_preview"])

    rules_html = ""

    for rule, records in results["retail_rule_findings"].items():

        readable_name = rule.replace("_", " ").title()

        rules_html += f"""
            <div class="rule-card">

                <h3>{html.escape(readable_name)}</h3>

                <h2>{len(records):,}</h2>

                <p>Records flagged for review</p>

                {make_table(records, 5)}

            </div>
        """

    if not rules_html:
        rules_html = "<p>No applicable retail rules found.</p>"

    cleaning_html = ""

    if cleaning_preview is not None:

        summary = cleaning_preview["summary"]

        cleaning_html = f"""
        <div class="section">

            <h2>7. Proposed Cleaning Results</h2>

            <p>
                These are preview results only.
                The original dataset has not been modified.
            </p>

            <div class="kpi-grid">

                <div class="kpi">
                    <h3>Original Records</h3>
                    <p>{summary["original_records"]:,}</p>
                </div>

                <div class="kpi">
                    <h3>Records Removed</h3>
                    <p>{summary["records_removed"]:,}</p>
                </div>

                <div class="kpi">
                    <h3>Records Quarantined</h3>
                    <p>{summary["records_quarantined"]:,}</p>
                </div>

                <div class="kpi">
                    <h3>Clean Preview</h3>
                    <p>{summary["records_after_preview"]:,}</p>
                </div>

            </div>

            <h3>Proposed Clean Dataset</h3>

            {make_table(cleaning_preview["cleaned_preview"])}

            <h3>Quarantined Records</h3>

            {make_table(cleaning_preview["quarantine_preview"])}

            <div class="notice">
                Review these records before approving the cleaning operation.
            </div>

            <form method="POST"
                  action="/approve/{html.escape(filename)}">

                <input type="hidden"
                       name="duplicate_action"
                       value="{html.escape(summary["duplicate_action"])}">

                <input type="hidden"
                       name="missing_action"
                       value="{html.escape(summary["missing_action"])}">

                <p>
                    By approving, you authorize FlowForge to export
                    the proposed cleaned and quarantined datasets.
                </p>

                <button type="submit">
                    Approve Cleaning and Export
                </button>

            </form>

        </div>
        """

    content = f"""
        <h1>FlowForge</h1>

        <h2>Retail Data Quality Dashboard</h2>

        <p class="subtitle">
            Dataset: {html.escape(filename)}
        </p>

        <div class="kpi-grid">

            <div class="kpi">
                <h3>Total Records</h3>
                <p>{results["total_rows"]:,}</p>
            </div>

            <div class="kpi">
                <h3>Total Columns</h3>
                <p>{results["total_columns"]}</p>
            </div>

            <div class="kpi">
                <h3>Missing Cells</h3>
                <p>{results["missing_cells"]:,}</p>
            </div>

            <div class="kpi">
                <h3>Missing Records</h3>
                <p>{results["missing_rows"]:,}</p>
            </div>

            <div class="kpi">
                <h3>Duplicates</h3>
                <p>{results["duplicate_count"]:,}</p>
            </div>

            <div class="kpi">
                <h3>Completeness</h3>
                <p>{results["completeness"]}%</p>
            </div>

        </div>

        <div class="notice">
            <strong>Data preservation:</strong>
            FlowForge does not automatically modify or delete
            uploaded records.
        </div>

        <div class="section">
            <h2>1. Original Dataset Preview</h2>
            {original_preview}
        </div>

        <div class="section">
            <h2>2. Missing Value Analysis</h2>
            {missing_table}
        </div>

        <div class="section">

            <h2>3. Duplicate Analysis</h2>

            <p>
                Duplicate records:
                <strong>{results["duplicate_count"]:,}</strong>
            </p>

            {duplicate_table}

        </div>

        <div class="section">
            <h2>4. Data Type Analysis</h2>
            {type_table}
        </div>

        <div class="section">

            <h2>5. Retail Business Rule Validation</h2>

            <p>
                These records have been flagged for review.
                They have not been modified.
            </p>

            <div class="rule-grid">
                {rules_html}
            </div>

            <p>
                Negative quantities may indicate returns.
                Zero prices may indicate promotional transactions.
                These are warnings, not necessarily errors.
            </p>

        </div>

        <div class="section">

            <h2>6. Cleaning Recommendation and Preview</h2>

            <form class="form-box"
                  method="POST"
                  action="/clean/{html.escape(filename)}">

                <label>Duplicate handling</label>

                <select name="duplicate_action">

                    <option value="keep">Keep duplicates</option>

                    <option value="remove">
                        Remove duplicate copies
                    </option>

                    <option value="quarantine">
                        Move duplicates to quarantine
                    </option>

                </select>

                <label>Missing value handling</label>

                <select name="missing_action">

                    <option value="keep">Keep missing values</option>

                    <option value="remove">
                        Remove records with missing values
                    </option>

                    <option value="quarantine">
                        Move records to quarantine
                    </option>

                </select>

                <button type="submit">
                    Generate Cleaning Preview
                </button>

            </form>

        </div>

        {cleaning_html}

        <div class="section">

            <h2>8. Anomaly Detection</h2>

            <p>
                Run Isolation Forest, Local Outlier Factor
                and One-Class SVM on the uploaded dataset.
            </p>

            <p>
                Anomaly detection flags unusual records
                for investigation. It does not delete them.
            </p>

            <a class="button"
               href="/analyze/{html.escape(filename)}">
                Run Anomaly Detection
            </a>

        </div>

        <a class="button" href="/upload">Upload Another Dataset</a>

        <a class="button" href="/sql-dashboard">SQL Analysis Dashboard</a>

        <a class="button" href="/">Home</a>
    """

    return page_template("FlowForge Dashboard", content)


# ============================================================
# 4. SQL DASHBOARD CHART GENERATION
# ============================================================

def create_sql_dashboard_charts(analysis):

    monthly = analysis["sql_monthly_sales.csv"]
    countries = analysis["sql_country_sales.csv"]
    products = analysis["sql_top_products.csv"]
    quality = analysis["sql_data_quality_summary.csv"]
    duplicates = analysis["sql_duplicate_summary.csv"]

    chart_paths = {}

    # --------------------------------------------------------
    # CHART 1: MONTHLY SALES TREND
    # --------------------------------------------------------

    chart_path = os.path.join(
        VISUALIZATION_FOLDER,
        "sql_monthly_sales.png"
    )

    fig, ax = plt.subplots(figsize=(11, 5))

    if not monthly.empty:

        monthly = monthly.copy()

        monthly["total_sales"] = pd.to_numeric(
            monthly["total_sales"],
            errors="coerce"
        )

        monthly = monthly.dropna(
            subset=["total_sales"]
        )

        ax.plot(
            monthly["sales_month"],
            monthly["total_sales"],
            marker="o",
            linewidth=2
        )

        ax.set_title("Monthly Retail Sales Trend")
        ax.set_xlabel("Month")
        ax.set_ylabel("Sales Value")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3)

    else:
        ax.text(0.5, 0.5, "No monthly sales data available",
                ha="center", va="center")

    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    chart_paths["monthly"] = "sql_monthly_sales.png"

    # --------------------------------------------------------
    # CHART 2: TOP 10 COUNTRIES
    # --------------------------------------------------------

    chart_path = os.path.join(
        VISUALIZATION_FOLDER,
        "sql_country_sales.png"
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    if not countries.empty:

        country_plot = countries.head(10).copy()

        country_plot["total_sales"] = pd.to_numeric(
            country_plot["total_sales"],
            errors="coerce"
        )

        country_plot = country_plot.dropna(
            subset=["total_sales"]
        )

        country_plot = country_plot.sort_values(
            "total_sales"
        )

        ax.barh(
            country_plot["Country"].astype(str),
            country_plot["total_sales"]
        )

        ax.set_title("Top 10 Countries by Sales")
        ax.set_xlabel("Sales Value")
        ax.set_ylabel("Country")
        ax.grid(axis="x", alpha=0.3)

    else:
        ax.text(0.5, 0.5, "No country sales data available",
                ha="center", va="center")

    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    chart_paths["countries"] = "sql_country_sales.png"

    # --------------------------------------------------------
    # CHART 3: TOP 10 PRODUCTS
    # --------------------------------------------------------

    chart_path = os.path.join(
        VISUALIZATION_FOLDER,
        "sql_top_products.png"
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    if not products.empty:

        product_plot = products.head(10).copy()

        product_plot["total_sales"] = pd.to_numeric(
            product_plot["total_sales"],
            errors="coerce"
        )

        product_plot = product_plot.dropna(
            subset=["total_sales"]
        )

        product_plot["Product"] = (
            product_plot["StockCode"].astype(str)
            + " - "
            + product_plot["Description"].fillna("").astype(str)
        )

        product_plot = product_plot.sort_values(
            "total_sales"
        )

        ax.barh(
            product_plot["Product"],
            product_plot["total_sales"]
        )

        ax.set_title("Top 10 Products by Sales")
        ax.set_xlabel("Sales Value")
        ax.set_ylabel("Product")
        ax.grid(axis="x", alpha=0.3)

    else:
        ax.text(0.5, 0.5, "No product data available",
                ha="center", va="center")

    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    chart_paths["products"] = "sql_top_products.png"

    # --------------------------------------------------------
    # CHART 4: DATA QUALITY ISSUES
    # --------------------------------------------------------

    chart_path = os.path.join(
        VISUALIZATION_FOLDER,
        "sql_data_quality.png"
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    if not quality.empty:

        quality_row = quality.iloc[0]

        labels = [
            "Missing Customer ID",
            "Missing Description",
            "Returns",
            "Invalid Prices",
            "Duplicate Rows"
        ]

        duplicate_count = (
            int(duplicates.iloc[0]["duplicate_records"])
            if not duplicates.empty else 0
        )

        values = [
            int(quality_row["missing_customer_id"]),
            int(quality_row["missing_description"]),
            int(quality_row["return_records"]),
            int(quality_row["invalid_price_records"]),
            duplicate_count
        ]

        ax.bar(labels, values)

        ax.set_title("Retail Data Quality Findings")
        ax.set_ylabel("Number of Records")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)

    else:
        ax.text(0.5, 0.5, "No quality data available",
                ha="center", va="center")

    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    chart_paths["quality"] = "sql_data_quality.png"

    return chart_paths


# ============================================================
# 5. HOME PAGE
# ============================================================

@app.route("/")
def home():

    content = """
        <h1>FlowForge</h1>

        <h2>Retail ETL and Data Quality Platform</h2>

        <p>
            Upload your retail dataset to perform
            data profiling, cleaning and anomaly detection.
        </p>

        <div class="section">

            <h2>Platform Capabilities</h2>

            <ul>
                <li>Multi-format data ingestion</li>
                <li>Data quality analysis</li>
                <li>Missing value and duplicate detection</li>
                <li>Retail business-rule validation</li>
                <li>Approval-based data cleaning</li>
                <li>Quarantine management</li>
                <li>Machine learning anomaly detection</li>
                <li>SQL staging and final loading</li>
                <li>SQL data-quality analysis</li>
                <li>Audit logging and data exports</li>
            </ul>

            <a class="button" href="/upload">
                Start Processing
            </a>

            <a class="button" href="/sql-dashboard">
                View SQL Analysis Dashboard
            </a>

            <a class="button" href="/model-evaluation-dashboard">
                View Model Evaluation Dashboard
            </a>

        </div>
    """

    return page_template("FlowForge Home", content)


# ============================================================
# 6. SQL ANALYSIS DASHBOARD
# ============================================================

@app.route("/sql-dashboard")
def sql_dashboard():

    if not os.path.isfile(DATABASE_PATH):

        return error_page(
            "FlowForge database not found. "
            "Please run the SQL pipeline first."
        )

    try:

        # Run the existing SQL queries and export reports.
        analysis = export_sql_analysis()

        overview = analysis["sql_dataset_overview.csv"]
        quality = analysis["sql_data_quality_summary.csv"]
        duplicates = analysis["sql_duplicate_summary.csv"]
        monthly = analysis["sql_monthly_sales.csv"]
        countries = analysis["sql_country_sales.csv"]
        products = analysis["sql_top_products.csv"]

        if overview.empty:
            raise ValueError("SQL dataset overview returned no data.")

        overview_row = overview.iloc[0]
        quality_row = quality.iloc[0]

        chart_paths = create_sql_dashboard_charts(analysis)

        duplicate_count = (
            int(duplicates.iloc[0]["duplicate_records"])
            if not duplicates.empty else 0
        )

        total_records = int(overview_row["total_records"])

        missing_customer = int(
            quality_row["missing_customer_id"]
        )

        missing_description = int(
            quality_row["missing_description"]
        )

        return_records = int(
            quality_row["return_records"]
        )

        invalid_prices = int(
            quality_row["invalid_price_records"]
        )

        content = f"""
            <h1>FlowForge</h1>

            <h2>SQL Analysis Dashboard</h2>

            <p class="subtitle">
                SQL-based analysis of the retail transactions
                stored in the FlowForge SQLite database.
            </p>

            <div class="notice">

                <strong>Database:</strong>
                flowforge.db

                <br>

                <strong>Analysis:</strong>
                SQL queries executed against retail_transactions.

                <br>

                <strong>Records:</strong>
                {total_records:,}

            </div>

            <div class="kpi-grid">

                <div class="kpi">
                    <h3>Total Records</h3>
                    <p>{format_number(total_records)}</p>
                </div>

                <div class="kpi">
                    <h3>Total Invoices</h3>
                    <p>{format_number(overview_row["total_invoices"])}</p>
                </div>

                <div class="kpi">
                    <h3>Unique Products</h3>
                    <p>{format_number(overview_row["unique_products"])}</p>
                </div>

                <div class="kpi">
                    <h3>Unique Customers</h3>
                    <p>{format_number(overview_row["unique_customers"])}</p>
                </div>

                <div class="kpi">
                    <h3>Countries</h3>
                    <p>{format_number(overview_row["countries"])}</p>
                </div>

            </div>

            <div class="section">

                <h2>1. Dataset Overview</h2>

                {make_table(overview)}

            </div>

            <div class="section">

                <h2>2. Data Quality Analysis</h2>

                <div class="kpi-grid">

                    <div class="kpi">
                        <h3>Missing Customer IDs</h3>
                        <p>{format_number(missing_customer)}</p>
                    </div>

                    <div class="kpi">
                        <h3>Missing Descriptions</h3>
                        <p>{format_number(missing_description)}</p>
                    </div>

                    <div class="kpi">
                        <h3>Return Records</h3>
                        <p>{format_number(return_records)}</p>
                    </div>

                    <div class="kpi">
                        <h3>Invalid Prices</h3>
                        <p>{format_number(invalid_prices)}</p>
                    </div>

                    <div class="kpi">
                        <h3>Duplicate Rows</h3>
                        <p>{format_number(duplicate_count)}</p>
                    </div>

                </div>

                {make_table(quality)}

                <h3>Duplicate Analysis</h3>

                {make_table(duplicates)}

                <p>
                    Returns and zero or negative prices are flagged
                    for review. They are not automatically treated
                    as invalid transactions.
                </p>

            </div>

            <div class="section">

                <h2>3. Monthly Sales Trend</h2>

                <img
                    class="chart"
                    src="/sql-dashboard-chart/{chart_paths["monthly"]}"
                    alt="Monthly sales trend"
                >

                <h3>Monthly SQL Results</h3>

                {make_table(monthly, limit=30)}

            </div>

            <div class="section">

                <h2>4. Country-Wise Sales Analysis</h2>

                <img
                    class="chart"
                    src="/sql-dashboard-chart/{chart_paths["countries"]}"
                    alt="Country-wise sales chart"
                >

                <h3>Country Sales Results</h3>

                {make_table(countries, limit=38)}

            </div>

            <div class="section">

                <h2>5. Top 10 Products</h2>

                <img
                    class="chart"
                    src="/sql-dashboard-chart/{chart_paths["products"]}"
                    alt="Top products by sales"
                >

                <h3>Top Product Results</h3>

                {make_table(products, limit=10)}

            </div>

            <div class="section">

                <h2>6. Data Quality Findings Visualization</h2>

                <img
                    class="chart"
                    src="/sql-dashboard-chart/{chart_paths["quality"]}"
                    alt="Data quality findings"
                >

            </div>

            <div class="section">

                <h2>7. Download SQL Reports</h2>

                <p>
                    <a class="button"
                       href="/download/sql_dataset_overview.csv">
                        Dataset Overview
                    </a>
                </p>

                <p>
                    <a class="button"
                       href="/download/sql_data_quality_summary.csv">
                        Data Quality Report
                    </a>
                </p>

                <p>
                    <a class="button"
                       href="/download/sql_duplicate_summary.csv">
                        Duplicate Report
                    </a>
                </p>

                <p>
                    <a class="button"
                       href="/download/sql_monthly_sales.csv">
                        Monthly Sales Report
                    </a>
                </p>

                <p>
                    <a class="button"
                       href="/download/sql_country_sales.csv">
                        Country Sales Report
                    </a>
                </p>

                <p>
                    <a class="button"
                       href="/download/sql_top_products.csv">
                        Top Products Report
                    </a>
                </p>

            </div>

            <a class="button" href="/">Home</a>

            <a class="button" href="/upload">
                Upload Dataset
            </a>

            <a class="button" href="/model-evaluation-dashboard">
                Model Evaluation Dashboard
            </a>

        """

        return page_template(
            "FlowForge SQL Analysis Dashboard",
            content
        )

    except Exception as error:

        return error_page(
            "SQL dashboard could not be generated: " + str(error)
        )


# ============================================================
# 7. SQL DASHBOARD CHART IMAGE ROUTE
# ============================================================

@app.route("/sql-dashboard-chart/<filename>")
def sql_dashboard_chart(filename):

    allowed_charts = {
        "sql_monthly_sales.png",
        "sql_country_sales.png",
        "sql_top_products.png",
        "sql_data_quality.png"
    }

    if filename not in allowed_charts:
        return error_page("Invalid dashboard chart.")

    filepath = os.path.join(
        VISUALIZATION_FOLDER,
        filename
    )

    if not os.path.isfile(filepath):
        return error_page("Dashboard chart not found.")

    return send_from_directory(
        VISUALIZATION_FOLDER,
        filename
    )


# ============================================================
# 8. MODEL EVALUATION KPI DASHBOARD
# ============================================================

@app.route("/model-evaluation-dashboard")
def model_evaluation_dashboard():

    dashboard_folder = VISUALIZATION_FOLDER

    dashboard_file = "flowforge_model_evaluation_dashboard.png"

    dashboard_path = os.path.join(
        dashboard_folder,
        dashboard_file
    )

    if not os.path.isfile(dashboard_path):

        return error_page(
            "Model evaluation dashboard not found. "
            "Please run model_evaluation_dashboard.py first."
        )

    content = """
        <h1>FlowForge</h1>

        <h2>Model Evaluation KPI Dashboard</h2>

        <p class="subtitle">
            Performance comparison of Isolation Forest,
            Local Outlier Factor, One-Class SVM and
            Majority Voting Ensemble.
        </p>

        <div class="section">

            <img
                src="/model-evaluation-dashboard-image"
                alt="FlowForge Model Evaluation KPI Dashboard"
                style="
                    width:100%;
                    height:auto;
                    border-radius:8px;
                "
            >

        </div>

        <a class="button" href="/sql-dashboard">
            SQL Analysis Dashboard
        </a>

        <a class="button" href="/upload">
            Upload Dataset
        </a>

        <a class="button" href="/">Home</a>
    """

    return page_template(
        "Model Evaluation Dashboard",
        content
    )


@app.route("/model-evaluation-dashboard-image")
def model_evaluation_dashboard_image():

    dashboard_folder = VISUALIZATION_FOLDER

    dashboard_file = "flowforge_model_evaluation_dashboard.png"

    dashboard_path = os.path.join(
        dashboard_folder,
        dashboard_file
    )

    if not os.path.isfile(dashboard_path):
        return error_page("Dashboard image not found.")

    return send_from_directory(
        dashboard_folder,
        dashboard_file
    )


# ============================================================
# 9. DATASET UPLOAD
# ============================================================

@app.route("/upload", methods=["GET", "POST"])
def upload_file():

    if request.method == "GET":

        content = """
            <h1>FlowForge</h1>

            <h2>Upload Your Dataset</h2>

            <p>Supported formats: CSV, XLSX and JSON</p>

            <div class="section">

                <form method="POST" enctype="multipart/form-data">

                    <input type="file"
                           name="file"
                           accept=".csv,.xlsx,.json"
                           required>

                    <br>

                    <button type="submit">
                        Upload and Inspect
                    </button>

                </form>

            </div>

            <a class="button" href="/">Home</a>
            <a class="button" href="/sql-dashboard">SQL Dashboard</a>
        """

        return page_template("Upload Dataset", content)

    if "file" not in request.files:
        return error_page("No file selected.")

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        return error_page("Please select a dataset.")

    if not allowed_file(uploaded_file.filename):
        return error_page("Unsupported file format.")

    original_name = secure_filename(uploaded_file.filename)

    extension = original_name.rsplit(".", 1)[1].lower()

    filename = f"{uuid.uuid4().hex[:8]}_{original_name}"

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:

        uploaded_file.save(filepath)

        df = read_dataset(filepath, extension)

        if df.empty or len(df.columns) == 0:
            raise ValueError("The uploaded dataset is empty.")

        analysis = run_data_quality_analysis(df)

        analysis["dataset_preview"] = df.head(10).copy()

        return dashboard_page(filename, analysis)

    except Exception as error:

        if os.path.exists(filepath):
            os.remove(filepath)

        return error_page(error)


# ============================================================
# 10. CLEANING PREVIEW
# ============================================================

@app.route("/clean/<filename>", methods=["POST"])
def preview_cleaning(filename):

    filename = secure_filename(filename)

    try:

        df = read_uploaded_file(filename)

        duplicate_action = request.form.get(
            "duplicate_action", "keep"
        )

        missing_action = request.form.get(
            "missing_action", "keep"
        )

        validate_cleaning_actions(
            duplicate_action,
            missing_action
        )

        cleaning_preview = build_cleaning_preview(
            df,
            duplicate_action=duplicate_action,
            missing_action=missing_action
        )

        analysis = run_data_quality_analysis(df)

        analysis["dataset_preview"] = df.head(10).copy()

        return dashboard_page(
            filename,
            analysis,
            cleaning_preview
        )

    except Exception as error:

        return error_page(error)


# ============================================================
# 11. APPROVE CLEANING AND EXPORT
# ============================================================

@app.route("/approve/<filename>", methods=["POST"])
def approve_cleaning(filename):

    filename = secure_filename(filename)

    try:

        original_df = read_uploaded_file(filename)

        duplicate_action = request.form.get(
            "duplicate_action", "keep"
        )

        missing_action = request.form.get(
            "missing_action", "keep"
        )

        validate_cleaning_actions(
            duplicate_action,
            missing_action
        )

        cleaning_preview = build_cleaning_preview(
            original_df,
            duplicate_action=duplicate_action,
            missing_action=missing_action
        )

        cleaned_df = cleaning_preview["cleaned_preview"]

        quarantine_df = cleaning_preview["quarantine_preview"]

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        cleaned_filename = f"cleaned_data_{timestamp}.csv"

        quarantine_filename = f"quarantined_data_{timestamp}.csv"

        cleaned_path = os.path.join(
            EXPORT_FOLDER,
            cleaned_filename
        )

        quarantine_path = os.path.join(
            EXPORT_FOLDER,
            quarantine_filename
        )

        cleaned_df.to_csv(cleaned_path, index=False)

        quarantine_df.to_csv(quarantine_path, index=False)

        audit_record = {
            "original_filename": filename,
            "approval_datetime": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "duplicate_action": duplicate_action,
            "missing_action": missing_action,
            "original_records": len(original_df),
            "cleaned_records": len(cleaned_df),
            "quarantined_records": len(quarantine_df),
            "cleaned_file": cleaned_filename,
            "quarantined_file": quarantine_filename
        }

        audit_df = pd.DataFrame([audit_record])

        if os.path.exists(AUDIT_LOG):

            audit_df.to_csv(
                AUDIT_LOG,
                mode="a",
                header=False,
                index=False
            )

        else:

            audit_df.to_csv(
                AUDIT_LOG,
                index=False
            )

        content = f"""
            <h1>FlowForge</h1>

            <div class="success">

                <h2>Cleaning Approved Successfully</h2>

                <p>Your original uploaded dataset remains unchanged.</p>

            </div>

            <div class="section">

                <h2>Export Summary</h2>

                <p>Original records: {len(original_df):,}</p>

                <p>Cleaned records: {len(cleaned_df):,}</p>

                <p>Quarantined records: {len(quarantine_df):,}</p>

                <h2>Download Your Files</h2>

                <p>
                    <a class="button"
                       href="/download/{cleaned_filename}">
                        Download Cleaned Dataset
                    </a>
                </p>

                <p>
                    <a class="button"
                       href="/download/{quarantine_filename}">
                        Download Quarantined Dataset
                    </a>
                </p>

                <p>
                    <a class="button"
                       href="/download/audit_log.csv">
                        Download Audit Log
                    </a>
                </p>

            </div>

            <a class="button" href="/upload">Upload Another Dataset</a>

            <a class="button" href="/sql-dashboard">SQL Dashboard</a>

            <a class="button" href="/">Home</a>
        """

        return page_template("Cleaning Approved", content)

    except Exception as error:

        return error_page(error)


# ============================================================
# 12. ANOMALY DETECTION
# ============================================================

@app.route("/analyze/<filename>")
def analyze_dataset(filename):

    filename = secure_filename(filename)

    try:

        df = read_uploaded_file(filename)

        print("Dataset shape:", df.shape)
        print("Total records:", len(df))
        print("Dataset filename:", filename)

        original_data, X_scaled, feature_names = prepare_anomaly_data(df)

        results_df, execution_summary = run_all_anomaly_models(
            original_data,
            X_scaled
        )

        exported_files = export_anomaly_results(
            results_df,
            execution_summary,
            EXPORT_FOLDER
        )

        anomaly_count = int(
            results_df["Combined_Anomaly"].sum()
        )

        total_records = len(results_df)

        anomaly_rate = (
            anomaly_count / total_records * 100
            if total_records > 0 else 0
        )

        execution_table = make_table(
            execution_summary,
            limit=20
        )

        anomaly_preview = make_table(
            results_df[
                results_df["Combined_Anomaly"] == 1
            ].sort_values(
                "Models_Flagged",
                ascending=False
            ),
            limit=20
        )

        content = f"""
            <h1>FlowForge</h1>

            <h2>Anomaly Detection Results</h2>

            <p>
                Dataset: {html.escape(filename)}
            </p>

            <div class="kpi-grid">

                <div class="kpi">
                    <h3>Records Analysed</h3>
                    <p>{total_records:,}</p>
                </div>

                <div class="kpi">
                    <h3>Anomalies Detected</h3>
                    <p>{anomaly_count:,}</p>
                </div>

                <div class="kpi">
                    <h3>Anomaly Rate</h3>
                    <p>{anomaly_rate:.2f}%</p>
                </div>

                <div class="kpi">
                    <h3>Models Executed</h3>
                    <p>3</p>
                </div>

            </div>

            <div class="section">

                <h2>1. Model Execution Summary</h2>

                {execution_table}

            </div>

            <div class="section">

                <h2>2. Detected Anomalies</h2>

                <p>
                    Records are ranked by the number of models
                    that flagged them.
                </p>

                {anomaly_preview}

            </div>

            <div class="section">

                <h2>3. Download Results</h2>

                <p>
                    <a class="button"
                       href="/download/{os.path.basename(exported_files["complete_results"])}">
                        Download Complete Analysis
                    </a>
                </p>

                <p>
                    <a class="button"
                       href="/download/{os.path.basename(exported_files["detected_anomalies"])}">
                        Download Detected Anomalies
                    </a>
                </p>

                <p>
                    <a class="button"
                       href="/download/{os.path.basename(exported_files["execution_summary"])}">
                        Download Model Execution Summary
                    </a>
                </p>

            </div>

            <div class="warning">

                <strong>Important:</strong>

                Anomaly detection identifies unusual patterns.
                Anomalies are not automatically errors or invalid
                transactions. Review records before taking action.

            </div>

            <a class="button" href="/upload">Upload Another Dataset</a>

            <a class="button" href="/sql-dashboard">SQL Dashboard</a>

            <a class="button" href="/">Home</a>
        """

        return page_template("Anomaly Detection Results", content)

    except Exception as error:

        return error_page(error)


# ============================================================
# 13. FILE DOWNLOAD
# ============================================================

@app.route("/download/<filename>")
def download_file(filename):

    filename = secure_filename(filename)

    filepath = os.path.join(EXPORT_FOLDER, filename)

    if not os.path.isfile(filepath):
        return error_page("Requested export file was not found.")

    return send_from_directory(
        EXPORT_FOLDER,
        filename,
        as_attachment=True
    )


# ============================================================
# 14. ERROR HANDLING
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return (
        "File is too large. Maximum upload size is 200 MB.",
        413
    )


@app.errorhandler(404)
def page_not_found(error):

    return error_page("The requested page was not found.")


# ============================================================
# 15. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
        
    )
