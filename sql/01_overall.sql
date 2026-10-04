-- Overall churn: total customers, churners and churn rate.
-- Reads the raw CSV relative to the repository root. Churn is read as text because DuckDB would
-- otherwise auto-detect the Yes/No column as BOOLEAN.
WITH customers AS (
    SELECT
        *,
        TRY_CAST(NULLIF(TRIM("TotalCharges"), '') AS DOUBLE) AS total_charges,
        CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END AS churned
    FROM read_csv_auto('WA_Fn-UseC_-Telco-Customer-Churn.csv', types = {'Churn': 'VARCHAR'})
)
SELECT
    COUNT(*) AS customers,
    SUM(churned) AS churners,
    AVG(churned) AS churn_rate
FROM customers;
