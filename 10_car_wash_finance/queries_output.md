# SQL answers

Written by `run_queries.py` from `queries.sql`. All data is made up.

## 1. Rank the sites by 2025 profit margin, and show each one's share of chain profit

| rank | name | region | revenue | profit | margin_pct | share_of_chain_profit_pct |
|---|---|---|---|---|---|---|
| 1 | Site 01 | North | 2,495,325 | 1,276,006 | 51.1 | 7.6 |
| 2 | Site 03 | North | 2,465,760 | 1,244,045 | 50.5 | 7.4 |
| 3 | Site 24 | South | 1,886,954 | 888,117 | 47.1 | 5.3 |
| 4 | Site 21 | South | 2,114,696 | 974,491 | 46.1 | 5.8 |
| 5 | Site 15 | Central | 1,946,518 | 882,709 | 45.3 | 5.3 |
| 6 | Site 23 | South | 1,839,069 | 819,916 | 44.6 | 4.9 |
| 7 | Site 04 | North | 1,986,423 | 871,644 | 43.9 | 5.2 |
| 8 | Site 06 | North | 1,834,252 | 780,792 | 42.6 | 4.7 |
| 9 | Site 20 | South | 1,809,444 | 769,920 | 42.6 | 4.6 |
| 10 | Site 17 | Central | 1,628,715 | 680,481 | 41.8 | 4.1 |
| 11 | Site 02 | North | 1,835,219 | 762,437 | 41.5 | 4.5 |
| 12 | Site 13 | Central | 1,870,321 | 773,477 | 41.4 | 4.6 |
| 13 | Site 12 | Central | 1,901,749 | 777,634 | 40.9 | 4.6 |
| 14 | Site 16 | Central | 1,897,149 | 761,526 | 40.1 | 4.5 |
| 15 | Site 25 | South | 1,663,372 | 640,007 | 38.5 | 3.8 |
| 16 | Site 08 | North | 1,495,774 | 549,867 | 36.8 | 3.3 |
| 17 | Site 22 | South | 1,475,161 | 529,981 | 35.9 | 3.2 |
| 18 | Site 11 | Central | 1,391,541 | 495,644 | 35.6 | 3.0 |
| 19 | Site 05 | North | 1,409,108 | 454,780 | 32.3 | 2.7 |
| 20 | Site 07 | North | 1,731,871 | 555,384 | 32.1 | 3.3 |
| 21 | Site 09 | North | 1,286,555 | 399,734 | 31.1 | 2.4 |
| 22 | Site 19 | South | 1,110,332 | 287,147 | 25.9 | 1.7 |
| 23 | Site 10 | Central | 1,222,165 | 306,700 | 25.1 | 1.8 |
| 24 | Site 14 | Central | 1,134,645 | 239,139 | 21.1 | 1.4 |
| 25 | Site 27 | South | 782,157 | 47,432 | 6.1 | 0.3 |
| 26 | Site 26 | South | 824,173 | 13,019 | 1.6 | 0.1 |
| 27 | Site 18 | Central | 768,095 | 354.0 | 0.0 | 0.0 |

## 2. Chain revenue by month, against the month before and the same month a year earlier

| month | revenue | vs_last_month_pct | vs_last_year_pct |
|---|---|---|---|
| 2025-12 | 3,680,520 | -1.1 | 8.2 |
| 2025-11 | 3,722,666 | -1.1 | 10.3 |
| 2025-10 | 3,763,360 | 2.0 | 13.5 |
| 2025-09 | 3,688,069 | 1.9 | 11.8 |
| 2025-08 | 3,619,741 | -0.2 | 13.1 |
| 2025-07 | 3,625,980 | 0.0 | 10.2 |
| 2025-06 | 3,625,063 | -0.1 | 12.5 |
| 2025-05 | 3,628,070 | -2.6 | 10.0 |
| 2025-04 | 3,726,363 | 0.8 | 10.8 |
| 2025-03 | 3,697,938 | 5.2 | 6.8 |
| 2025-02 | 3,514,123 | -0.0 | 6.8 |
| 2025-01 | 3,514,651 | 3.4 | 5.3 |

## 3. Early warning: sites whose cancel rate over the last three months is well above the chain's

| name | opened | competitors | cancel_rate_pct | chain_cancel_rate_pct | times_the_chain |
|---|---|---|---|---|---|
| Site 27 | 2025-04 | 1 | 5.59 | 2.99 | 1.87 |
| Site 11 | 2015-08 | 2 | 4.56 | 2.99 | 1.53 |

## 4. Of members who joined in each quarter, what share were still members 3, 6 and 12 months later?

| quarter | joined | still_here_after_3_pct | still_here_after_6_pct | still_here_after_12_pct |
|---|---|---|---|---|
| 2023 Q1 | 6858 | 80.6 | 72.4 | 58.2 |
| 2023 Q2 | 7332 | 80.4 | 72.1 | 58.5 |
| 2023 Q3 | 6972 | 81.0 | 72.8 | 58.4 |
| 2023 Q4 | 6912 | 80.5 | 71.9 | 57.6 |
| 2024 Q1 | 7259 | 80.5 | 72.3 | 57.3 |
| 2024 Q2 | 7177 | 80.3 | 71.9 | 56.8 |
| 2024 Q3 | 6846 | 80.3 | 71.5 | 57.0 |
| 2024 Q4 | 6927 | 79.8 | 70.8 | 56.9 |
| 2025 Q1 | 6874 | 78.1 | 70.3 |  |
| 2025 Q2 | 7794 | 80.9 | 72.4 |  |
| 2025 Q3 | 7603 | 79.7 |  |  |
| 2025 Q4 | 7905 |  |  |  |

## 5. The five sites that missed their 2025 profit budget by the most

| name | budget_profit | actual_profit | difference | of_which_revenue |
|---|---|---|---|---|
| Site 11 | 665,204 | 495,644 | -169,560 | -242,548 |
| Site 03 | 1,352,668 | 1,244,045 | -108,623 | -91,177 |
| Site 01 | 1,368,364 | 1,276,006 | -92,358 | -99,265 |
| Site 16 | 848,883 | 761,526 | -87,357 | -40,927 |
| Site 21 | 1,061,445 | 974,491 | -86,953 | -81,360 |
