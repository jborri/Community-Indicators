An Ordinary Least Squares Regression Analysis will help us understand the relationship between chosen indicators and a sense of community. An OLS Model will also attempt to explain the variation of our dependent variable. 

Dependent Variable: Social Support
o Selected from the World Happiness Index, which defines Social Support as the following:
§ "Social support (or having someone to count on in times of trouble) is the national average of the binary responses (either 0 or 1) to the GWP question “If you were in trouble, do you have relatives or friends you can count on to help you whenever you need them, or not?”
o The Reasoning behind why this variable was chosen is because the WHI collected data from a similar number of countries that were also analyzed by the other 3 reports, allowing us to better explore the relationship between Social Support and the other indicators. And more importantly, WHI’s definition of Social Support is exactly the kind of metric I would like to explore; how might the other indicators affect how close you are to those around you? What might influence your ability to effectively rely on your community for support?

OLS Regression Results Without MPI
![alt text](<OLS Model.png>)

Correlation of Indicators without MPI
![alt text](Corr.png)

OLS Regression Results *with* MPI
![alt text](<OLS with Poverty.png>)

Correlation of Indicators *with* MPI
![alt text](Corr_Poverty.png)

Scatter Plot of Indicators with the Lowest P-Value
![alt text](OLS_SS_CSN.png)
![alt text](OLS_SS_LL.png)

Matrix of OLS Model without Poverty
![alt text](OLS_Matrix.png)

Matrix of OLS Model *with* Poverty
![alt text](OLS_Matrix_Poverty.png)


# Analysis of OLS Regression Results

I decided to run the OLS model twice, once with the indicator for Multidimensional Poverty and once without. My reasoning for this is because not only did the MP indicator have a large P- Value, but it also reduced the F-Statistic Significantly. I did see the value of having this indicator as a sort of control variable that prevented the influence of other stronger indicators on the model. The effects of the MP indicator can be seen in the matrix graphs above.
For the model without the MP Indicator, because the probability of the F-Statistic is lower
than .05, at a value of 1.43e-43 we can assume this test is statistically significant and the results are reliable. The Goodness of Fit of the model, represented by the R-Squared value is 0.835. This means that 83.5% of the variation of Social Support is accounted for in the model. Community Safety Net and HDI had the highest coefficient values of 0.5256 and 0.3005, respectively, meaning they have the greatest effect on Social Support. The only indicator with a Probability of T-Test value greater than .05 is the Life Ladder indicator. I found this interesting because as a measurement of happiness, it is the least statistically significant indicator. I feel this might have something to do with the intangible and hard-to-measure nature of feelings and emotions thus resulting in a weaker relationship. Another thing I found interesting was that the Tolerance for Immigrants indicator had a negative coefficient, meaning that as tolerance increases, a sense of community decreases. I feel as though this result might have to do with differing values and cultures that initially can prevent forming bonds and interacting.
With the MP indicator the probability of F-Statistic is still lower than .05 at a value of 1.47e-19 so the results are still reliable. The R-Squared value was reduced to .792, accounting for 79.2% of the variation. It also had a negative value for the coefficient and a very high Probability for the t-test at 0.872. I found this interesting because it might mean that even though social
support decreases as poverty increases, there is no relationship between community and whether or not the said community is in poverty.
