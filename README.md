# FlowForge
## Retail ETL and Data-Quality Anomaly Detection Pipeline

### 1. Project Overview

FlowForge is a Python-based retail data processing and analytics platform designed to perform data ingestion, profiling, preprocessing, data-quality validation, anomaly detection, SQL-based analysis and result export.

The system processes retail transaction data through an integrated ETL pipeline and applies machine learning techniques to identify unusual transaction patterns.

### 2. Project Objectives

- Ingest and process retail transaction datasets.
- Perform data profiling and exploratory data analysis.
- Identify missing values, duplicate records and potential data-quality issues.
- Apply data-cleaning rules with user approval.
- Maintain cleaned and quarantined data outputs.
- Detect unusual transaction patterns using machine learning.
- Evaluate anomaly detection models using precision, recall and F1-score.
- Store and analyse transaction data using SQLite.
- Generate visualisations and downloadable analytical reports.

### 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualisation | Matplotlib, Seaborn |
| Web Framework | Flask |
| Database | SQLite |
| Development Environment | VS Code, Jupyter Notebook |

### 4. System Modules

**Module 1 – Data Ingestion**

Accepts retail transaction datasets for processing.

**Module 2 – Data Profiling**

Analyses dataset dimensions, column types, missing values, duplicates and data-quality indicators.

**Module 3 – Data Cleaning and Validation**

Provides configurable handling of duplicates and missing values, identifies potential quality issues and supports approval-based cleaning.

**Module 4 – Anomaly Detection**

Implements three machine learning algorithms:

- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM

The system combines model predictions to identify transactions requiring further review.

**Module 5 – Model Evaluation**

Evaluates anomaly detection performance using:

- Precision
- Recall
- F1-score

Synthetic anomalies with known labels are used to evaluate model performance.

**Module 6 – SQL Analysis**

Stores retail transactions in SQLite and executes SQL queries for dataset overview, data quality, monthly sales, country-wise sales and top products.

**Module 7 – Visualisation and Export**

Generates analytical charts and provides downloadable datasets and reports.

### 5. Dataset

The project uses the Online Retail dataset.

| Attribute | Value |
|---|---:|
| Total records | 541,909 |
| Total columns | 8 |
| Missing cells | 136,534 |
| Duplicate records | 5,268 |
| Countries | 38 |
| Unique products | 4,070 |
| Unique customers | 4,372 |

The dataset contains retail transaction information such as invoice number, product code, description, quantity, invoice date, unit price, customer ID and country.

### 6. Anomaly Detection Results

The anomaly detection pipeline was executed on all 541,909 records.

| Metric | Result |
|---|---:|
| Records analysed | 541,909 |
| Combined anomalies detected | 8,041 |
| Anomaly rate | 1.48% |
| Models executed | 3 |

The individual models identify different numbers of unusual transactions. The combined output ranks transactions according to the number of models flagging them.

Anomalies indicate unusual patterns and should be reviewed before any business action is taken.

### 7. SQL Analysis

The SQLite database is named `flowforge.db`, with the primary transaction table `retail_transactions`.

The SQL dashboard provides:

- Dataset overview
- Data-quality analysis
- Duplicate analysis
- Monthly sales trends
- Country-wise sales analysis
- Top 10 products by sales
- Downloadable SQL reports

### 8. Running the Project

Clone or open the project directory and install the dependencies.

```bash
pip install -r requirements.txt
```

Start the Flask application:

```bash
python app.py
```

Open the application in a browser:

```text
http://127.0.0.1:5000
```

### 9. Expected Outputs

The system generates:

- Dataset profiling results
- Cleaning and validation results
- Cleaned transaction data
- Quarantined transaction data, where applicable
- Audit logs
- Anomaly detection results
- Model evaluation results
- SQL analytical reports
- Visualisation charts

### 10. Limitations

- Anomaly detection identifies unusual patterns, not confirmed fraud.
- Synthetic anomaly labels are used for model evaluation because confirmed real-world anomaly labels are not available.
- Some missing customer IDs, returns and zero-price transactions may represent legitimate retail business cases.
- Model performance may vary depending on dataset characteristics and parameter settings.

### 11. Conclusion

FlowForge integrates data engineering, data-quality validation, machine learning and SQL analytics into a unified retail transaction processing platform. It provides a structured approach to analysing retail data, identifying unusual patterns and generating useful analytical outputs.

**FlowForge — Retail ETL and Data Quality Platform**

Developed as a Data Science and Machine Learning project.
