# FlowForge-A-Multi-Source-ETL-Data-Quality-Visualization-and-Anomaly-Detection-Platform-Using-Python\
# FlowForge

## A Multi-Source ETL, Data Quality, Visualization and Anomaly Detection Platform Using Python

FlowForge is a Python-based data preparation and analytics platform inspired by tools such as Alteryx. It is designed to help users connect to different data sources, clean and transform data, validate data quality, detect anomalies, generate visualizations, and export processed datasets through a simple interactive interface.

The project focuses on building a lightweight real-world ETL platform that combines data engineering, data quality monitoring, machine learning, and visualization in one workflow.

---

## Project Objectives

The main objectives of FlowForge are to:

- Support multiple data sources such as CSV, Excel, JSON, SQLite databases, and REST APIs.
- Provide automated data cleaning and transformation features.
- Perform data quality validation on uploaded datasets.
- Identify missing values, duplicates, invalid records, and inconsistent data.
- Calculate an overall data quality score.
- Detect unusual or anomalous transactions using machine learning.
- Provide useful visualizations for understanding data distributions and quality issues.
- Compare data quality before and after cleaning.
- Export the final cleaned and validated dataset.

---

## Key Features

### Multi-Source Data Ingestion
FlowForge supports data ingestion from:

- CSV
- Excel
- JSON
- SQLite
- REST API

Additional database connections such as PostgreSQL and MySQL can be added as future enhancements.

### Data Preparation

Users can perform operations such as:

- Column selection
- Column renaming
- Filtering
- Missing-value handling
- Duplicate removal
- Data type conversion
- Basic calculated columns
- Data transformation

### Data Quality Validation

The platform checks for:

- Missing values
- Duplicate records
- Invalid data types
- Invalid or inconsistent values
- Outliers
- Data completeness
- Data uniqueness
- Data validity

A data quality score is generated based on these checks.

### Anomaly Detection

FlowForge uses machine-learning techniques such as **Isolation Forest** to identify unusual records or transactions based on selected numeric features.

### Data Visualization

The platform provides visualizations such as:

- Missing-value analysis
- Numeric distributions
- Category distributions
- Outlier visualizations
- Anomaly plots
- Before vs after data-quality comparison

### Data Export

Users can export the cleaned and validated dataset for further analysis or reporting.

---

## Project Workflow

```text
Data Sources
     |
     v
CSV / Excel / JSON / SQLite / API
     |
     v
Data Ingestion
     |
     v
Data Preview
     |
     v
Data Cleaning
     |
     v
Data Transformation
     |
     v
Data Quality Validation
     |
     v
Anomaly Detection
     |
     v
Visualization
     |
     v
Clean Dataset Export
