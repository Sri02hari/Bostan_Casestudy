#!/usr/bin/env python
# coding: utf-8

# <center>
#     <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/Logos/organization_logo/organization_logo.png" width="300" alt="cognitiveclass.ai logo">
# </center>
# 

# #### Import the required libraries we need for the lab.
# 

# In[16]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols


# #### Read the dataset in the csv file from the URL
# 

# In[4]:


boston_df=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv')


# #### Add your code below following the instructions given in the course to complete the peer graded assignment
# 

# In[6]:


boston_df.head(5)


# # TASK 2

# In[10]:



# Boxplot for the "Median value of owner-occupied homes"
plt.figure(figsize=(10, 6))
sns.boxplot(y=boston_df['MEDV'])
plt.title('Boxplot of Median Value of Owner-Occupied Homes')
plt.ylabel('MEDV ($1000\'s)')
plt.show()


# Boxplot for MEDV vs AGE (discretized)
boston_df['AGE_GROUP'] = pd.cut(boston_df['AGE'], bins=[0, 35, 70, 100], labels=['0-35', '35-70', '70+'])
plt.figure(figsize=(10, 6))
sns.boxplot(x='AGE_GROUP', y='MEDV', data=boston_df)
plt.title('Boxplot of MEDV vs AGE Group')
plt.xlabel('AGE Group')
plt.ylabel('MEDV ($1000\'s)')
plt.show()


# **Inferences for Boxplot of Median Value of Owner-Occupied Homes**
# 
# 1. Central Tendency and Spread:
# Median home value is around $21,000, with most homes ranging between $17,000 and $25,000.
# 
# 2. Outliers:
# There are several high-value outliers above $35,000, indicating some homes are significantly more expensive.
# 
# **Inferences for Boxplot of MEDV vs AGE Group**
# 1. Age Group Comparison:
# Newer homes (0-35 years) generally have higher values compared to older homes.
# 
# 2. Variability:
# Newer homes show more variation in value, with several high-value outliers, while older homes have more consistent values.

# In[14]:



# Bar plot for the Charles River variable
plt.figure(figsize=(10, 6))
sns.countplot(x=boston_df['CHAS'])
plt.title('Bar Plot of Charles River Variable')
plt.xlabel('CHAS (1 if tract bounds river; 0 otherwise)')
plt.ylabel('Count')
plt.show()


# **Inferences for Bar Plot of Charles River Variable**
# 
# 1. Distribution of Properties:
# The majority of properties do not bound the Charles River, indicating that most of the homes in the dataset are located away from the river.
# 
# 2. Proximity to River:
# Only a small number of properties are near the river, which could imply that riverfront properties are less common and potentially more exclusive or valuable.
# 

# In[12]:



# Scatter plot for NOX vs INDUS
plt.figure(figsize=(10, 6))
sns.scatterplot(x='NOX', y='INDUS', data=boston_df)
plt.title('Scatter Plot of NOX vs INDUS')
plt.xlabel('NOX (Nitric Oxides Concentration)')
plt.ylabel('INDUS (Proportion of Non-Retail Business Acres)')
plt.show()


# **Inferences for Scatter Plot of NOX vs INDUS**
# 
# 1. Positive Correlation: There is a positive correlation between NOX (Nitric Oxides Concentration) and INDUS (Proportion of Non-Retail Business Acres). As the concentration of NOX increases, the proportion of non-retail business acres also tends to increase.
# 
# 2. Clustered Data Points: The data points are more densely clustered at lower NOX concentrations (around 0.4 to 0.5), indicating that most areas have lower NOX levels and varying proportions of non-retail business acres. However, there are some outliers with higher NOX concentrations and higher INDUS values.

# In[13]:



# Histogram for the pupil to teacher ratio variable
plt.figure(figsize=(10, 6))
sns.histplot(boston_df['PTRATIO'], bins=10, kde=True)
plt.title('Histogram of Pupil to Teacher Ratio')
plt.xlabel('Pupil-Teacher Ratio')
plt.ylabel('Frequency')
plt.show()


# **Inference for Histogram Pupil to Teacher**
# 
# 1. High Frequency at Higher Ratios: The histogram shows that the pupil-to-teacher ratio is most frequently around 20, with the highest frequency bar indicating that many schools have a ratio close to this value.
# 
# 2. Distribution Spread: The distribution of the pupil-to-teacher ratio is skewed towards higher values, with fewer schools having lower ratios (around 14-16) and a gradual increase in frequency as the ratio approaches 20, followed by a sharp decline after 20.

# # TASK 3

# In[17]:



# 1. T-test for Independent Samples
# Hypothesis: Is there a significant difference in median value of houses bounded by the Charles river or not?

# Grouping data based on Charles River variable
group1 = boston_df[boston_df['CHAS'] == 1]['MEDV']
group2 = boston_df[boston_df['CHAS'] == 0]['MEDV']

# Perform T-test
t_stat, p_value = stats.ttest_ind(group1, group2)

print("T-test for Independent Samples")
print(f"T-statistic: {t_stat}, P-value: {p_value}")

# Conclusion
if p_value < 0.05:
    print("Reject the null hypothesis: There is a significant difference in the median value of houses bounded by the Charles River.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in the median value of houses bounded by the Charles River.")


# # Inference
# 
# **T-test for Independent Samples**
# 
# 1. Significant Difference in Median Values: The test results show a significant difference in the median value of houses near the Charles River compared to those not near the river, with a very low p-value (0.0000739). This means that the difference is not due to random chance.
# 2. Higher Median Values Near the River: Houses located near the Charles River generally have higher median values. This could be because of the attractive location and scenic views that make these properties more desirable.

# In[18]:



# 2. ANOVA
# Hypothesis: Is there a difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE)?

# Perform ANOVA
model = ols('MEDV ~ C(AGE)', data=boston_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("\nANOVA")
print(anova_table)

# Conclusion
if anova_table["PR(>F)"][0] < 0.05:
    print("Reject the null hypothesis: There is a significant difference in the median values of houses across different AGE groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in the median values of houses across different AGE groups.")


# # Inference
# 
# **ANOVA**
# 
# 1. No Significant Difference Across AGE Groups: The ANOVA test results indicate that there is no significant difference in the median values of houses based on the proportion of owner-occupied units built before 1940. The p-value (0.55397) is much higher than the 0.05 threshold, suggesting that the age of the houses does not play a major role in determining their median values.
# 2. Uniform Median Values: This implies that whether a house was built before or after 1940 does not significantly affect its median value. The median values of houses remain fairly consistent regardless of their age.

# In[19]:



# 3. Pearson Correlation
# Hypothesis: Can we conclude that there is no relationship between Nitric oxide concentrations and proportion of non-retail business acres per town?

# Perform Pearson Correlation
corr, p_value = stats.pearsonr(boston_df['NOX'], boston_df['INDUS'])

print("\nPearson Correlation")
print(f"Correlation coefficient: {corr}, P-value: {p_value}")

# Conclusion
if p_value < 0.05:
    print("Reject the null hypothesis: There is a significant relationship between NOX and INDUS.")
else:
    print("Fail to reject the null hypothesis: There is no significant relationship between NOX and INDUS.")


# # Inference
# 
# **Pearson Correlation**
# 
# 1. Significant Positive Relationship: The Pearson correlation results show a strong positive relationship between Nitric Oxide concentrations (NOX) and the proportion of non-retail business acres per town (INDUS), with a very high correlation coefficient (0.7637) and an extremely low p-value (7.91e-98). This means that as the proportion of industrial areas increases, the NOX levels also increase.
# 2. Higher NOX with More Industrial Areas: Towns with a higher proportion of industrial areas tend to have higher Nitric Oxide concentrations, likely due to pollution from industrial activities. This highlights the environmental impact of industrial zones on air quality.

# In[20]:



# 4. Regression Analysis
# Hypothesis: What is the impact of an additional weighted distance to the five Boston employment centres on the median value of owner occupied homes?

# Perform Regression Analysis
X = boston_df['DIS']
y = boston_df['MEDV']
X = sm.add_constant(X)  # adding a constant

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

print("\nRegression Analysis")
print(model.summary())

# Conclusion
if model.pvalues[1] < 0.05:
    print("Reject the null hypothesis: DIS has a significant impact on MEDV.")
else:
    print("Fail to reject the null hypothesis: DIS has no significant impact on MEDV.")


# # Inference 
# 
# **Regression Analysis**
# 
# 1. Significant Impact of Distance on Median Values: The regression analysis shows that the distance to the five Boston employment centres (DIS) significantly impacts the median value of owner-occupied homes, with a very low p-value (1.21e-08). This means that the distance to employment centers is an important factor in determining home values.
# 2. Positive Relationship: The positive coefficient (1.0916) indicates that as the distance to employment centers increases, the median value of homes also tends to increase. This could be because homes further from the city center might be larger, in more desirable suburban areas, or offer a better quality of life, making them more valuable.
