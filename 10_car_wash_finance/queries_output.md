# SQL answers

Written by `run_queries.py` from `queries.sql`. All data is made up.

## 1. Rank the sites by 2025 profit margin, and show each one's share of chain profit

| rank | name | region | revenue | profit | margin_pct | share_of_chain_profit_pct |
|---|---|---|---|---|---|---|
| 1 | Site 03 | North | 2,290,678 | 1,095,311 | 47.8 | 8.6 |
| 2 | Site 06 | North | 2,084,911 | 974,977 | 46.8 | 7.7 |
| 3 | Site 01 | North | 2,062,401 | 922,261 | 44.7 | 7.2 |
| 4 | Site 19 | South | 1,692,840 | 754,989 | 44.6 | 5.9 |
| 5 | Site 13 | Central | 1,939,094 | 843,931 | 43.5 | 6.6 |
| 6 | Site 16 | Central | 1,926,039 | 835,502 | 43.4 | 6.6 |
| 7 | Site 12 | Central | 1,843,602 | 765,834 | 41.5 | 6.0 |
| 8 | Site 02 | North | 1,717,237 | 680,486 | 39.6 | 5.3 |
| 9 | Site 10 | Central | 1,594,071 | 625,230 | 39.2 | 4.9 |
| 10 | Site 07 | North | 2,002,442 | 784,807 | 39.2 | 6.2 |
| 11 | Site 21 | South | 1,651,602 | 629,783 | 38.1 | 4.9 |
| 12 | Site 04 | North | 1,693,490 | 634,443 | 37.5 | 5.0 |
| 13 | Site 05 | North | 1,586,542 | 584,108 | 36.8 | 4.6 |
| 14 | Site 11 | Central | 1,472,360 | 538,154 | 36.6 | 4.2 |
| 15 | Site 09 | Central | 1,299,359 | 398,095 | 30.6 | 3.1 |
| 16 | Site 08 | North | 1,236,196 | 367,638 | 29.7 | 2.9 |
| 17 | Site 14 | Central | 1,253,502 | 354,956 | 28.3 | 2.8 |
| 18 | Site 20 | South | 1,198,478 | 326,549 | 27.2 | 2.6 |
| 19 | Site 15 | Central | 1,270,296 | 341,927 | 26.9 | 2.7 |
| 20 | Site 17 | South | 1,024,981 | 212,611 | 20.7 | 1.7 |
| 21 | Site 22 | South | 883,995 | 87,528 | 9.9 | 0.7 |
| 22 | Site 18 | South | 783,097 | -614.0 | -0.1 | -0.0 |
| 23 | Site 23 | South | 613,715 | -16,905 | -2.8 | -0.1 |

## 2. Chain revenue by month, against the month before and the same month a year earlier

| month | revenue | vs_last_month_pct | vs_last_year_pct |
|---|---|---|---|
| 2025-12 | 3,009,865 | 1.7 | 7.6 |
| 2025-11 | 2,959,573 | -0.4 | 10.5 |
| 2025-10 | 2,970,168 | -0.1 | 13.2 |
| 2025-09 | 2,972,835 | 4.2 | 12.3 |
| 2025-08 | 2,852,135 | -0.2 | 9.7 |
| 2025-07 | 2,857,809 | -2.1 | 10.0 |
| 2025-06 | 2,919,895 | -1.5 | 12.6 |
| 2025-05 | 2,965,400 | -0.1 | 13.1 |
| 2025-04 | 2,967,292 | 0.4 | 11.8 |
| 2025-03 | 2,954,075 | 3.1 | 5.0 |
| 2025-02 | 2,864,275 | 1.3 | 3.6 |
| 2025-01 | 2,827,609 | 1.1 | 6.5 |

## 3. Early warning: sites whose cancel rate over the last three months is well above the chain's

| name | opened | competitors | cancel_rate_pct | chain_cancel_rate_pct | times_the_chain |
|---|---|---|---|---|---|
| Site 23 | 2025-04 | 2 | 4.73 | 2.98 | 1.59 |
| Site 11 | 2017-01 | 2 | 4.3 | 2.98 | 1.44 |

## 4. Of members who joined in each quarter, what share were still members 3, 6 and 12 months later?

| quarter | joined | still_here_after_3_pct | still_here_after_6_pct | still_here_after_12_pct |
|---|---|---|---|---|
| 2023 Q1 | 5417 | 80.2 | 71.8 | 57.8 |
| 2023 Q2 | 6081 | 80.4 | 72.1 | 58.7 |
| 2023 Q3 | 5860 | 81.0 | 73.0 | 58.4 |
| 2023 Q4 | 5620 | 80.6 | 72.0 | 57.1 |
| 2024 Q1 | 5724 | 81.1 | 73.2 | 59.6 |
| 2024 Q2 | 5888 | 79.7 | 71.1 | 56.8 |
| 2024 Q3 | 5645 | 80.0 | 71.6 | 58.2 |
| 2024 Q4 | 5559 | 80.3 | 71.7 | 57.3 |
| 2025 Q1 | 5719 | 78.2 | 69.4 |  |
| 2025 Q2 | 6178 | 80.2 | 71.5 |  |
| 2025 Q3 | 6165 | 79.6 |  |  |
| 2025 Q4 | 6191 |  |  |  |

## 5. The five sites that missed their 2025 profit budget by the most

| name | budget_profit | actual_profit | difference | of_which_revenue |
|---|---|---|---|---|
| Site 11 | 727,506 | 538,154 | -189,352 | -251,808 |
| Site 06 | 1,041,068 | 974,977 | -66,091 | -109,168 |
| Site 12 | 829,338 | 765,834 | -63,504 | -64,056 |
| Site 02 | 737,977 | 680,486 | -57,491 | -69,029 |
| Site 03 | 1,151,481 | 1,095,311 | -56,170 | -87,123 |
