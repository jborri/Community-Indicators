A Logistic Regression Analysis will allow us to estimate whether or not these indicators are able predict what is considered a “strong sense of community” (Our dependent variable) and what isn’t. We will be dividing the indicators into the categories, Health, Economic/Poverty, Governmental, and Happiness/Communal, to increase the number of indicators to study, but to also see which one has a greater impact.

The Threshold for the Social Support Binary used was 0.73971. Any country with a value lower than this would be assigned as having a “weaker” sense of community. The value used relates to the 25% of countries that fall below this value. I used 25% because with such a small standard deviation it would be harder to identify any difference at a higher threshold. A similar number of indicators was used for each categorical model.

![alt text](Thresholdv.png)
![alt text](<Binary Count.png>)

Health Logit Regression Results:
![alt text](<Health Logit.png>)

Health Marginal Effects:
![alt text](<Health Marginal Effects.png>)

**Percentage of Correct Predictions of Health Model**
High Social Support: 83.8%
Low Social Support: 73.1%

Economic/Poverty Logit Regression Results:
![alt text](<Econ:Poverty Logit.png>)

Economic/Poverty Marginal Effects:
![alt text](<Poverty Marg effects.png>)

**Percentage of Correct Predictions of Economic/Poverty Model**
High Social Support: 84.5%
Low Social Support: 84.0%

Governmental Logit Regression Results:
![alt text](<Govt Logit.png>)

Government Marginal Effects:
![alt text](<Govt marg effects.png>)

**Percentage of Correct Predictions of Government Model**
High Social Support: 94.4%
Low Social Support: 73.3%

Happiness/Communal Logit Regression Results:
![alt text](<Happiness Logit.png>)

Happiness/Communal Marginal Effects:
![alt text](<Happy marg Effects.png>)

**Percentage of Correct Predictions of Happiness Model**
High Social Support: 92.2%
Low Social Support: 65.4%



Based off the prediction values, Happiness/Communal and Governmental had the highest values for predicting whether if a community had a high Social Support value, and also had the lowest values of predicting if they had a low social support value. These results coupled with the results of the marginal effects seem to align. Positive valued coefficients were much stronger than the negative valued ones. The Economic/Poverty model and the Health Model seemed to be the most reliable of the four for both predicting low and high social support values, and both had among the highest pseudo-R-squared values sitting around 30%.