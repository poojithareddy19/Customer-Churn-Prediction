-- Churn rate by contract type and payment method, with each segment's share of all churners.
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
    "Contract" AS contract,
    "PaymentMethod" AS payment_method,
    COUNT(*) AS customers,
    SUM(churned) AS churners,
    AVG(churned) AS churn_rate,
    SUM(churned) / SUM(SUM(churned)) OVER () AS share_of_all_churners
FROM customers
GROUP BY "Contract", "PaymentMethod"
ORDER BY churn_rate DESC;
