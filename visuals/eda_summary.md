# EDA summary

## Key insights
- Overall attrition rate is 15.9% (239 of 1500 employees), so the classes are imbalanced.
- Sales has the highest attrition (24.9%); Engineering has the lowest (10.5%).
- Employees working overtime leave at 23.0% versus 12.9% without overtime (1.8x higher).
- Attrition falls as job satisfaction rises: 30.8% at score 1 versus 9.5% at score 5.
- Median monthly income is ₹59,100 for leavers versus ₹77,000 for stayers (-23%).
- Employees with up to 1 year at the company leave at 34.8%, compared with 11.7% for those with 5+ years.
- Top factors by combined correlation / mutual-information rank: MonthlyIncome, Age, Department_Sales, YearsAtCompany, JobSatisfaction.

## Attrition by department
| Department   |   employees |   leavers | attrition_rate   |
|:-------------|------------:|----------:|:-----------------|
| Sales        |         437 |       109 | 24.9%            |
| Marketing    |         186 |        28 | 15.1%            |
| HR           |         119 |        16 | 13.4%            |
| Finance      |         279 |        30 | 10.8%            |
| Engineering  |         449 |        47 | 10.5%            |

## Attrition by overtime
| OverTime   |   employees |   leavers | attrition_rate   |
|:-----------|------------:|----------:|:-----------------|
| Yes        |         448 |       103 | 23.0%            |
| No         |        1052 |       136 | 12.9%            |

## Attrition by job satisfaction
|   JobSatisfaction |   employees |   leavers | attrition_rate   |
|------------------:|------------:|----------:|:-----------------|
|                 1 |         130 |        40 | 30.8%            |
|                 2 |         269 |        67 | 24.9%            |
|                 3 |         431 |        61 | 14.2%            |
|                 4 |         401 |        42 | 10.5%            |
|                 5 |         220 |        21 | 9.5%             |

## Factor ranking (EDA)
| feature                |   correlation |   mutual_info |   avg_rank |
|:-----------------------|--------------:|--------------:|-----------:|
| MonthlyIncome          |        -0.276 |         0.047 |      1.000 |
| Age                    |        -0.193 |         0.021 |      2.500 |
| Department_Sales       |         0.158 |         0.026 |      3.500 |
| YearsAtCompany         |        -0.172 |         0.013 |      4.500 |
| JobSatisfaction        |        -0.178 |         0.013 |      4.500 |
| WorkLifeBalance        |        -0.086 |         0.016 |      5.500 |
| Department_Finance     |        -0.068 |         0.009 |      8.500 |
| OverTime               |         0.126 |         0.000 |      8.750 |
| Department_Engineering |        -0.079 |         0.003 |      9.000 |
| DistanceFromHome       |         0.084 |         0.000 |      9.750 |
| PerformanceRating      |         0.002 |         0.008 |     10.500 |
| Department_HR          |        -0.020 |         0.000 |     11.250 |
| Department_Marketing   |        -0.009 |         0.000 |     11.750 |
