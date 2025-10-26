# Center for Disease Control and Prevention, Behavioral Risk Factor Surveillance System
## DSE 203 - Machine Learning
## Milestone 2 - Data Exploration and Initial Preprocessing

**Environment Setup Requirements**

This Jupyter Notebook is self-contained and can be run in most environments that support Jupyter Notebooks like Visual Studio Code or Google Colab.

**Data Exploration**
1. How many observations does your dataset have?

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
| Data_value | Percentage of Break_Out_Category and Break_Out make up total | Continuous | 1 - 100 | idk distribution |
| Confidence_limit_Low | Low Confidence Limit | Continous | idk scale | idk distribution |
| Confidence_limit_High | High Confidence Limit | Continous | idk scale | idk distribution |
| Display_order | Display order | Continous | idk scale | idk distribution |
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

*`DataSource` has only 'BRFSS'*

*`Data_Value_Footnote_Symbol` has only \*, \*\*, or \*\*\**

*`Data_value_unit` has only '%'*

*`Data_value_type` has only 'Crude Prevalence'.*

**Data Plots**

1. Plot your data with various types of charts like bar charts, pie charts, scatter plots etc. and clearly explain the plots. For image data, you will need to plot your example classes.
2. How will you preprocess your data? Handle data imbalance if needed. 




table of content(optional)
introduction
figures
preprocessing steps
Model
results
discussion
conclusions
future work:
statement of collaboration:



