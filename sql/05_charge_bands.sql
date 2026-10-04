-- Churn rate by monthly charge quintile (band 1 = lowest charges). customerID breaks ties so
-- customers with the same charge always land in the same band.
-- Reads the raw CSV relative to the repository root. Churn is read as text because DuckDB would
-- otherwise auto-detect the Yes/No column as BOOLEAN.
WITH customers AS (
    SELECT
        *,
        TRY_CAST(NULLIF(TRIM("TotalCharges"), '') AS DOUBLE) AS total_charges,
        CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END AS churned
    FROM read_csv_auto('WA_Fn-UseC_-Telco-Customer-Churn.csv', types = {'Churn': 'VARCHAR'})
)
, bands AS (
    SELECT
        churned,
        "MonthlyCharges" AS monthly_charges,
        NTILE(5) OVER (ORDER BY "MonthlyCharges", "customerID") AS charge_band
    FROM customers
)
SELECT
    charge_band,
    COUNT(*) AS customers,
    SUM(churned) AS churners,
    AVG(churned) AS churn_rate,
    AVG(monthly_charges) AS avg_monthly_charges,
    MIN(monthly_charges) AS min_monthly_charges,
    MAX(monthly_charges) AS max_monthly_charges
FROM bands
GROUP BY charge_band
ORDER BY charge_band;
