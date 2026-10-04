-- Monthly revenue lost to churners per contract type, largest first, with its cumulative share.
-- Reads the raw CSV relative to the repository root. Churn is read as text because DuckDB would
-- otherwise auto-detect the Yes/No column as BOOLEAN.
WITH customers AS (
    SELECT
        *,
        TRY_CAST(NULLIF(TRIM("TotalCharges"), '') AS DOUBLE) AS total_charges,
        CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END AS churned
    FROM read_csv_auto('WA_Fn-UseC_-Telco-Customer-Churn.csv', types = {'Churn': 'VARCHAR'})
)
, by_contract AS (
    SELECT
        "Contract" AS contract,
        COUNT(*) AS customers,
        SUM(churned) AS churners,
        AVG(churned) AS churn_rate,
        SUM(CASE WHEN churned = 1 THEN "MonthlyCharges" ELSE 0 END) AS churned_monthly_revenue
    FROM customers
    GROUP BY "Contract"
)
SELECT
    contract,
    customers,
    churners,
    churn_rate,
    churned_monthly_revenue,
    churned_monthly_revenue / SUM(churned_monthly_revenue) OVER () AS share_of_lost_revenue,
    SUM(churned_monthly_revenue) OVER (ORDER BY churned_monthly_revenue DESC)
        / SUM(churned_monthly_revenue) OVER () AS cumulative_share_of_lost_revenue
FROM by_contract
ORDER BY churned_monthly_revenue DESC;
