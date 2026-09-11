# Community Indicators

An exploratory cross-national analysis of how social, economic, health, and governance indicators are associated with reported social support.

[View the companion Tableau dashboard](https://public.tableau.com/app/profile/joseph.borri/viz/CommunityIndicators_17224364594760/Dashboard1?publish=yes)

![Relationship between social support and community safety net](OLS%20Regression%20Analysis/OLS_SS_CSN.png)

## Project question

What measurable conditions are associated with a stronger reported sense of social support across countries?

The project combines country-level indicators from four sources and examines the question through:

- ordinary least squares regression with social support as a continuous outcome;
- logistic regression after converting social support into a lower-support indicator; and
- exploratory visual analysis in Python and Tableau.

## Data sources

| Source | Reference year | Role in the analysis |
|---|---:|---|
| Social Progress Index | 2015 | Social and institutional indicators |
| United Nations Human Development Index | 2015 | Human-development measures |
| World Happiness Report | 2019 | Social support, life evaluation, and related measures |
| Global Multidimensional Poverty Index | 2015 | Multidimensional poverty measures |

Countries are aligned by name and merged across the source files included in the repository. Because the sources cover different years and definitions, the merged dataset is appropriate for exploratory association analysis rather than causal inference.

## Methods

### OLS analysis

The OLS model examines associations between social support and selected measures including:

- Human Development Index;
- Life Ladder;
- community safety net; and
- tolerance for immigrants.

The analysis compares specifications with and without the multidimensional-poverty measure and includes correlation plots and coefficient diagnostics.

### Logistic analysis

For the classification analysis, countries below the first quartile of observed social support (`0.739719`) are assigned to the lower-support class. Separate exploratory models examine grouped health, economic, governance, and happiness/community indicators.

Reported classification values in the existing analysis are in-sample summaries. They should not be interpreted as estimates of performance on unseen countries.

## Repository structure

```text
Community-Indicators/
├── Logistic Regression Analysis/
│   ├── LOGITFINAL.py
│   ├── source datasets
│   └── model outputs and figures
├── OLS Regression Analysis/
│   ├── OLSF.py
│   ├── source datasets
│   └── model outputs and figures
└── README.md
```

## Run locally

```bash
git clone https://github.com/jborri/Community-Indicators.git
cd Community-Indicators

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 "OLS Regression Analysis/OLSF.py"
python3 "Logistic Regression Analysis/LOGITFINAL.py"
```

Each script reads the data files from its own directory, so the project does not depend on a user-specific filesystem path.

## Selected findings

- Social support is positively associated with several development and community-safety measures in the assembled sample.
- The OLS specifications explain a substantial share of within-sample variation, but the result is sensitive to variable selection and overlapping constructs.
- The economic and health logistic models show more balanced in-sample classification than the governance and happiness/community specifications.

These findings are descriptive and model-dependent. They do not establish that any indicator causes stronger or weaker communities.

## Limitations

- Source datasets represent different years and measurement frameworks.
- Country-name alignment and missing data reduce the available sample differently across models.
- Several predictors are conceptually related, creating possible multicollinearity.
- The binary threshold is an analytical choice rather than a validated definition of community strength.
- The logistic results do not use a held-out evaluation set.

## Tools

Python, pandas, statsmodels, scikit-learn, SciPy, seaborn, Matplotlib, and Tableau.
