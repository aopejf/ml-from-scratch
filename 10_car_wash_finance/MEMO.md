# Memo: 2025 results and where to build site 24

**To:** Chief Financial Officer &nbsp; **From:** Financial analysis &nbsp; **About:** a made-up 23-site car wash chain

> Every number here comes from made-up data, written to show how I would work. No real company is involved.

## Recommendation

**Build site 24 at location B (the commuter road).** It is the only one of the three that clears our 10% required
return even if volume comes in low. Do not build at A, the busiest road: three competitors and the highest cost
leave it short of the bar.

| Location | Cars passing a day | Competitors | Build cost | Value created (NPV) | If volume is low | Yearly return (IRR) |
|---|---|---|---|---|---|---|
| A: Retail park | 41,000 | 3 | $4.6m | -$0.2m | -$1.4m | 9% |
| **B: Commuter road** | 34,000 | 1 | $4.1m | **+$1.5m** | **+$0.4m** | **15%** |
| C: Edge of town | 24,000 | 0 | $3.5m | +$0.8m | -$0.6m | 13% |

B needs about 130,500 washes a year to break even. Our 20 established sites predict 148,300. Across those sites,
each competitor within three miles costs about 13,400 washes a year, which is why A's traffic does not save it.

## Where 2025 landed

Revenue was $35.1m. Site profit was $12.7m (36% of revenue) and $9.0m after head-office cost. That is
**$864k below budget**. Three things caused it, and costs partly offset them:

| Cause | Effect on profit |
|---|---|
| Fewer members than planned (about 60,000 on average against 61,700) | -$637k |
| Fewer retail cars: the budget assumed 3% growth, established sites fell 2% | -$623k |
| Membership price: the $2 rise came in March, the budget assumed January | -$222k |
| Costs under budget, partly because fewer cars were washed | +$594k |

## Four sites need action

| Site | What is wrong | What I would do | Worth |
|---|---|---|---|
| 07 | Labour is 25% of revenue; sites of its volume run near 19% | Review the rota against car counts by hour | About $120k a year |
| 14 | Chemical cost per car jumped from $0.67 to $1.03 in June 2025 | Check dosing equipment for a fault or leak this week | About $37k a year |
| 11 | A competitor opened in September 2024. Members fell from 2,961 to 2,175 and the cancel rate rose from 2.9% to 4.6% a month | Test a retention offer on half of its members and measure the difference | It missed budget by $189k |
| 18 | Has lost money three years running on a quiet road with four competitors, while paying $158k a year in rent | Find out when the lease can be ended, and what it would cost | Stops the loss; frees management time |

Sites 22 and 23 look weak but opened in the last two years and are still filling up. They should be judged against
their opening plan, not against established sites.

## 2026 outlook

Forecast revenue is **$36.7m, up 4.4%**. Tested on 2025 using only earlier years, the forecast's monthly car counts
were off by 4.8% on average, a little better than assuming "same as this month last year" (5.7%).

## What I am less sure of

- The site-24 volume prediction rests on 20 sites. The low and high cases cover about 4 in 5 outcomes, not all of them.
- A new site may take customers from our nearby sites. That is not in the numbers and would lower B's value.
- About a third of the cash value of site 24 is the assumed sale value after ten years (six times yearly profit). With no sale value at all, B would fall $0.5m short of the 10% bar, though it still earns back its build cost in 7.1 years. The decision depends on believing the site is worth something in year ten.
- The value of a new member ($771 against $54 to win one) counts all marketing as the cost of winning members and nothing for opening discounts, so the true ratio is lower than 14 to 1.

*Workings: `analysis.py`, the Excel model `car_wash_finance_model.xlsx`, and the dashboard.*
