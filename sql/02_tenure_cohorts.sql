-- Churn rate by tenure cohort (months since sign-up).
-- Reads the raw CSV relative to the repository root. Churn is read as text because DuckDB would
-- otherwise auto-detect the Yes/No column as BOOLEAN.
WITH customers AS (
    SELECT
        *,
        TRY_CAST(NULLIF(TRIM("TotalCharges"), '') AS DOUBLE) AS total_charges,
        CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END AS churned
    FROM read_csv_auto('WA_Fn-UseC_-Telco-Customer-Churn.csv', types = {'Churn': 'VARCHAR'})
)
, cohorts AS (
    SELECT
        churned,
        CASE
            WHEN tenure <= 6 THEN '0-6'
            WHEN tenure <= 12 THEN '7-12'
            WHEN tenure <= 24 THEN '13-24'
            WHEN tenure <= 48 THEN '25-48'
            ELSE '49+'
        END AS tenure_cohort,
        CASE
            WHEN tenure <= 6 THEN 1
            WHEN tenure <= 12 THEN 2
            WHEN tenure <= 24 THEN 3
            WHEN tenure <= 48 THEN 4
            ELSE 5
        END AS cohort_order
    FROM customers
)
SELECT
    tenure_cohort,
    COUNT(*) AS customers,
    SUM(churned) AS churners,
    AVG(churned) AS churn_rate
FROM cohorts
GROUP BY tenure_cohort, cohort_order
ORDER BY cohort_order;
