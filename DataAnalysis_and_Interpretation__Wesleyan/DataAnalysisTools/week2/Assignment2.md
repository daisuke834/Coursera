#syntax used to run an ANOVA
[Link to Code Syntax](https://github.com/daisuke834/Coursera/blob/master/DataAnalysis_and_Interpretation__Wesleyan/DataAnalysisTools/week2/Assignment2_ChiSquare.py "Link to Code Syntax") <-Click Here

#output
[Link to Output](https://github.com/daisuke834/Coursera/blob/master/DataAnalysis_and_Interpretation__Wesleyan/DataAnalysisTools/week2/output.txt "Link to Output") <-Click Here

# interpretation
* Data: Adults age 20 to 50.
* Explanatory Variable: CURRENT MARITAL STATUS which is collapsed into 6 ordered categories. 1. Married, 2. Living with someone as if married, 3. Widowed, 4. Divorced, 5. Separated, 6. Never Married
* Response Variable: Alcohol abuse/dependence happen in the last 12 month

Chi-Square test revealed that current marital status and alcohol abuse were significantly associated, X2=180.5, p=4.2e-37.

Post hoc comparisons of alcohol abuse by pairs of marital statuses revealed that marital status and alcohol abuse were significantly associated for the following 5 comparisons among 15 comparisons because thse p-values were smaller than Bonferroni-Adjustment=0.0033:
* Married vs Living with someone (p=3.473738e-08)
* Married vs Divorced (p=0.000044)
* Married vs Separated (p=4.434827e-08)
* Married vs Never Married (p=1.625864e-39)
* Divorced vs Never Married (p=2.862612e-06)

The result showed that those who have been never married tend to have the higher rate of alcohol abuse than those who are married and those who are divorced. The result also showed that those who are married tend to have the lower rate of alcohol abuse than the others except for the widowed.

![BarChart](barchart.png)
