# Center for Disease Control and Prevention, Behavioral Risk Factor Surveillance System
## DSE 203 - Machine Learning
## Milestone 2 - Data Exploration and Initial Preprocessing
#### Dataset link - https://data.cdc.gov/Behavioral-Risk-Factors/Behavioral-Risk-Factor-Surveillance-System-BRFSS-P/dttw-5yxu

**Environment Setup Requirements**

This Jupyter Notebook is self-contained and can be run in most environments that support Jupyter Notebooks like Visual Studio Code or Google Colab.

**Data Exploration**
1. How many observations does your dataset have?
![Data_distribution_by_Class](visualizations/data_shape.png)
*Our dataest has 2,763,102 rows and 27 columns.*

2. Describe all columns in your dataset their scales and data distributions. Describe the categorical and continuous variables in your dataset. Describe your target column and if you are using images plot some example classes of the images.

| Column | Description | Categorical or Continuous | Scale | Distribution |
| --- | --- | --- | --- | --- |
| Year | Year of survey response | Continuous  | 2011 - 2023 | idk distribution |
| Locattionabbr | State abbreviation | Categorical | Nominal | idk distribution |
| Locationdesc | Full state name | Categorical | Nominal | idk distribution |
| Class | Class description for area of health | Categorical | Nominal | idk distribution |
| Topic | Subclass, specific area of interest | Categorical | Nominal | idk distribution |
| Question | Question in survey | Categorical | Nominal | idk distribution |
| Response | Response in survey | Categorical | Nominal | idk distribution |
| Break_Out | Demographic category value | Categorical | Loosely ordinal | idk distribution |
| Break_Out_Category | Demographic category value | Categorical | Nominal | idk distribution |
| Sample_Size | Size of demographic in Break_Out_Category and Break_Out | Continuous | idk scale | idk distribution |
| Data_value | Percentage of Break_Out_Category and Break_Out make up total | Continuous | 0 - 100 | Mean value is 40.45% with a median of 28.6%. 50% of the data is less than 28.6 |
| Confidence_limit_Low | Low Confidence Limit | Continous | 0 - 100 | Mean value is 37.10 with a median of 23.8. 50% of data less than 68.9 |
| Confidence_limit_High | High Confidence Limit | Continous | 0 - 100 | Mean value is 43.8 with a median of  33.3. 50% of the data is less than 33.3 |
| Display_order | Display order | Continous | 1 - 4493 | idk distribution |
| Data_value_unit | Unit for Data_value_type | Categorical | Nominal | All values are '%' |
| Data_value_type | Data value type, such as age-adjusted prevalence or crude prevalence | Categorical | Nominal | All values are 'crude prevalence' |
| 16  Data_Value_Footnote_Symbol | Symbol denoting footnote | Categorical | Nominal | distribution |
| Data_Value_Footnote | Footnote text | Categorical | Nominal | Mostly 'No responses for this data cell' |
| DataSource | Survey name | Categorical | Nominal | All values are 'BRFSS' |
| ClassId | Class ID | Categorical | Nominal | distribution |
| TopicId | Topic ID | Categorical | Nominal | distribution |
| LocationID | Location ID | Categorical | Nominal | distribution |
| BreakoutID | Breakout ID | Categorical | Nominal | distribution |
| BreakOutCategoryID | Breakout Category ID | Categorical | Nominal | distribution |
| QuestionID | Question ID | Categorical | Nominal | distribution |
| ResponseID | Response ID | Categorical | Nominal | distribution |
| GeoLocation | Latitude and longitude of state | Numeric | idk scale | idk distribution |

3. Do you have missing and duplicate values in your dataset? Note: For image data you can still describe your data by the number of classes, size of images, are sizes uniform? Do they need to be cropped? normalized? etc.

*Our dataset has a few columns that are entirely one value.*
![Data_distribution_by_Class](visualizations/Missing_value_code.png)

![Data_distribution_by_Class](visualizations/Missing_value_report.png)

*`DataSource` has only 'BRFSS'*

*`Data_Value_Footnote_Symbol` has only \*, \*\*, or \*\*\**

*`Data_value_unit` has only '%'*

*`Data_value_type` has only 'Crude Prevalence'.*

`Data_Value_Footnote` and `Data_Value_Footnote_Symbol` has around 80% missing values. `Confidence_limits` columns has aroud 20% missing values and `Response` and   `Geolocation` has less than 1% missing values.

Out dataset does not have any duplicated rows.

**Data Plots**

1. Plot your data with various types of charts like bar charts, pie charts, scatter plots etc. and clearly explain the plots. For image data, you will need to plot your example classes.
   
   (i) Total data distributions between classes
    ![Data_distribution_by_Class](visualizations/Data_distribution_by_Class.jpg)

   (ii) ![distribution_survey_responses](visualizations/distribution_survey_responses.jpg)

   Interpretation: The pie chart displays the top five survey responses, showing that “No” (≈50%) and “Yes” (≈48%) dominate the distribution, while “Not told they have arthritis,” “Good or Better Health,” and “Married” each contribute less than 2% of the total responses.

   (iii) ![confidence_limit_scatter_plot](visualizations/confidence_limit_scatter_plot.jpg)

   Interpretation: The scatter plot shows a strong positive relationship between the lower and upper confidence limits, with points closely following an upward trend. This indicates that as the lower limit increases, the upper limit also increases proportionally, and all upper limits remain above their corresponding lower limits, as expected.

   (iv) ![data_disribution_cancer](visualizations/data_disribution_cancer.jpg)

   (v) ![filtered_cancer_ds](visualizations/filtered_cancer_ds.jpg)

   (vi) ![top10_states_avg_health](visualizations/top10_states_avg_health.jpg)

   Interpretation: The bar plot showing the top 10 states by average health behavior rate. Each bar represents a state or territory, with the height indicating its mean Data_value (in percentage). The plot highlights that the U.S. Virgin Islands (VI) has the highest average, followed by Guam (GU) and Washington, D.C. (DC), while the other states like Delaware (DE), Wisconsin (WI), and Vermont (VT) have slightly lower but similar averages. Overall, it visually ranks these locations by their health behavior performance.




**How will you preprocess your data? Handle data imbalance if needed.**
- Handle missing values through imputation or removal.
- Dropping unnecessary columns
- Encode categorical variables using label or one-hot encoding.
- Scale numerical features if needed.
- Address class imbalance using resampling, class weighting or using any balancing algorithms suhch as SMOTE, ADASYN.
- Select relevant features to improve model performance.
  
**Future work**
Model predictions, clustering

**Statement of collaboration**




