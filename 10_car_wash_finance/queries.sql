-- Five questions a finance team would ask the database, answered in SQL.
-- Run them all with:  python run_queries.py   (results land in queries_output.md)
-- The database is data/chain.db (SQLite), written by make_data.py. All data is made up.


-- QUERY: 1. Rank the sites by 2025 profit margin, and show each one's share of chain profit
WITH year AS (
    SELECT site_id,
           SUM(membership_revenue + retail_revenue) AS revenue,
           SUM(membership_revenue + retail_revenue
               - labour - chemicals - utilities - maintenance - rent - marketing - card_fees - other) AS profit
    FROM monthly
    WHERE month LIKE '2025-%'
    GROUP BY site_id
)
SELECT RANK() OVER (ORDER BY profit / revenue DESC)        AS rank,
       s.name,
       s.region,
       ROUND(revenue)                                      AS revenue,
       ROUND(profit)                                       AS profit,
       ROUND(100.0 * profit / revenue, 1)                  AS margin_pct,
       ROUND(100.0 * profit / SUM(profit) OVER (), 1)      AS share_of_chain_profit_pct
FROM year
JOIN sites s USING (site_id)
ORDER BY rank;


-- QUERY: 2. Chain revenue by month, against the month before and the same month a year earlier
WITH chain AS (
    SELECT month, SUM(membership_revenue + retail_revenue) AS revenue
    FROM monthly
    GROUP BY month
)
SELECT month,
       ROUND(revenue)                                                              AS revenue,
       ROUND(100.0 * (revenue / LAG(revenue, 1)  OVER (ORDER BY month) - 1), 1)    AS vs_last_month_pct,
       ROUND(100.0 * (revenue / LAG(revenue, 12) OVER (ORDER BY month) - 1), 1)    AS vs_last_year_pct
FROM chain
ORDER BY month DESC
LIMIT 12;


-- QUERY: 3. Early warning: sites whose cancel rate over the last three months is well above the chain's
WITH rolling AS (
    SELECT site_id, month,
           SUM(cancels)       OVER three_months AS cancels_3m,
           SUM(members_start) OVER three_months AS members_3m
    FROM monthly
    WINDOW three_months AS (PARTITION BY site_id ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
),
latest AS (
    SELECT site_id, 1.0 * cancels_3m / members_3m AS cancel_rate
    FROM rolling
    WHERE month = (SELECT MAX(month) FROM monthly)
),
chain AS (
    SELECT 1.0 * SUM(cancels) / SUM(members_start) AS cancel_rate
    FROM monthly
    WHERE month >= '2025-10'
)
SELECT s.name,
       s.opened,
       s.competitors,
       ROUND(100.0 * l.cancel_rate, 2)                 AS cancel_rate_pct,
       ROUND(100.0 * c.cancel_rate, 2)                 AS chain_cancel_rate_pct,
       ROUND(l.cancel_rate / c.cancel_rate, 2)         AS times_the_chain
FROM latest l
JOIN sites s USING (site_id)
CROSS JOIN chain c
WHERE l.cancel_rate > 1.25 * c.cancel_rate
ORDER BY times_the_chain DESC;


-- QUERY: 4. Of members who joined in each quarter, what share were still members 3, 6 and 12 months later?
WITH joined AS (
    SELECT site_id, joined, active AS joined_count
    FROM cohorts
    WHERE months_later = 0
),
labelled AS (
    SELECT substr(c.joined, 1, 4) || ' Q' || ((CAST(substr(c.joined, 6, 2) AS INTEGER) + 2) / 3) AS quarter,
           c.months_later, c.active, j.joined_count
    FROM cohorts c
    JOIN joined j USING (site_id, joined)
)
SELECT quarter,
       SUM(CASE WHEN months_later = 0 THEN joined_count END)                                                                       AS joined,
       ROUND(100.0 * SUM(CASE WHEN months_later = 3  THEN active END) / SUM(CASE WHEN months_later = 3  THEN joined_count END), 1) AS still_here_after_3_pct,
       ROUND(100.0 * SUM(CASE WHEN months_later = 6  THEN active END) / SUM(CASE WHEN months_later = 6  THEN joined_count END), 1) AS still_here_after_6_pct,
       ROUND(100.0 * SUM(CASE WHEN months_later = 12 THEN active END) / SUM(CASE WHEN months_later = 12 THEN joined_count END), 1) AS still_here_after_12_pct
FROM labelled
GROUP BY quarter
ORDER BY quarter;


-- QUERY: 5. The five sites that missed their 2025 profit budget by the most
WITH actual AS (
    SELECT site_id,
           SUM(membership_revenue + retail_revenue) AS revenue,
           SUM(membership_revenue + retail_revenue
               - labour - chemicals - utilities - maintenance - rent - marketing - card_fees - other) AS profit
    FROM monthly
    WHERE month LIKE '2025-%'
    GROUP BY site_id
),
plan AS (
    SELECT site_id,
           SUM(membership_revenue + retail_revenue) AS revenue,
           SUM(membership_revenue + retail_revenue
               - labour - chemicals - utilities - maintenance - rent - marketing - card_fees - other) AS profit
    FROM budget
    GROUP BY site_id
)
SELECT s.name,
       ROUND(p.profit)                         AS budget_profit,
       ROUND(a.profit)                         AS actual_profit,
       ROUND(a.profit - p.profit)              AS difference,
       ROUND(a.revenue - p.revenue)            AS of_which_revenue
FROM actual a
JOIN plan p USING (site_id)
JOIN sites s USING (site_id)
ORDER BY difference
LIMIT 5;
