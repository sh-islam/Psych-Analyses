import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Generate dummy data with high correlations
np.random.seed(0)

# Creating a DataFrame with dummy data
data = {
    'Extraversion': np.random.rand(100),
    'Conscientiousness': np.random.rand(100),
    'Agreeableness': np.random.rand(100),
    'Neuroticism': np.random.rand(100),
    'Openness': np.random.rand(100),
}

# Create a high correlation between 'Openness' and 'Extraversion'
data['Extraversion'] = 0.8 * data['Openness'] + 0.2 * data['Extraversion']

# Create a high correlation between 'Openness' and 'Authoritative Leadership'
data['Authoritative Leadership'] = 0.7 * data['Openness'] + 0.3 * np.random.rand(100)

# Add other leadership traits
data['Transformational'] = np.random.rand(100)
data['Democratic'] = np.random.rand(100)
data['Laissez-Faire'] = np.random.rand(100)
data['Transactional'] = np.random.rand(100)
data['Servant'] = np.random.rand(100)

df = pd.DataFrame(data)

# Step 1: Calculate the correlation matrix of all variables
correlation_matrix_all = df.corr()
print("Correlation Matrix of All Variables:")
print(correlation_matrix_all)

# Step 2: Calculate the correlation matrix of just personality variables
personality_variables = df[['Extraversion', 'Conscientiousness', 'Agreeableness', 'Neuroticism', 'Openness']]
correlation_matrix_personality = personality_variables.corr()
print("\nCorrelation Matrix of Personality Variables:")
print(correlation_matrix_personality)

# Can't seem to use the same method as step 2; can't seem to use corrwith with two databases of uneven numbers, looks for pairwise correlations
# Step 3: Create a new DataFrame with personality variables and leadership traits
leadership_styles = df[['Authoritative Leadership', 'Transformational', 'Democratic', 'Laissez-Faire', 'Transactional', 'Servant']]
# Locate just personality and leadership correlations on the table
personality_leadership_correlations = correlation_matrix_all.loc[personality_variables.columns, leadership_styles.columns]
print("\nCorrelation Table between Personality Variables and Leadership Styles:")
print(personality_leadership_correlations)

# Step 4: Perform a multiple regression analysis
# For example, predicting 'Authoritative Leadership' based on 'Openness' and 'Extraversion'
X = df[['Openness', 'Extraversion']]
y = df['Authoritative Leadership']
X = sm.add_constant(X)  # Add a constant term (intercept) to the independent variables
model = sm.OLS(y, X).fit()  # Fit the multiple regression model
print("\nMultiple Regression Summary:")
print(model.summary())

# Multiple regression scatterplots:
# Scatter plot of 'Openness' vs. 'Authoritative Leadership' 
plt.scatter(df['Openness'], df['Authoritative Leadership'], label='Data Points', color='blue', alpha=0.5)
plt.xlabel('Openness')
plt.ylabel('Authoritative Leadership')
plt.title('Openness vs. Authoritative Leadership')
# Fit a regression line 
X_openness = df[['Openness']]
X_openness = sm.add_constant(X_openness)
model_openness = sm.OLS(y, X_openness).fit()
y_openness_pred = model_openness.predict(X_openness)
# Plot the regression line
plt.plot(df['Openness'], y_openness_pred, color='red', label='Regression Line')
# Add a legend
plt.legend()
# Save 
plt.savefig('openness_regression_plot.png')
plt.close()

plt.scatter(df['Extraversion'], df['Authoritative Leadership'], label='Data Points', color='green', alpha=0.5)
plt.xlabel('Extraversion')
plt.ylabel('Authoritative Leadership')
plt.title('Extraversion vs. Authoritative Leadership')
X_extraversion = df[['Extraversion']]
X_extraversion = sm.add_constant(X_extraversion)
model_extraversion = sm.OLS(y, X_extraversion).fit()
y_extraversion_pred = model_extraversion.predict(X_extraversion)
plt.plot(df['Extraversion'], y_extraversion_pred, color='red', label='Regression Line')
plt.legend()
plt.savefig('extraversion_regression_plot.png')
plt.close()

plt.scatter(df['Openness'], df['Authoritative Leadership'], label='Openness', color='blue', alpha=0.5)
plt.scatter(df['Extraversion'], df['Authoritative Leadership'], label='Extraversion', color='green', alpha=0.5)
plt.xlabel('Openness & Extraversion')
plt.ylabel('Authoritative Leadership')
plt.title('Combined Plot: Openness & Extraversion vs. Authoritative Leadership')
plt.plot(df['Openness'], y_openness_pred, color='red', linestyle='--', label='Openness Regression Line')
plt.plot(df['Extraversion'], y_extraversion_pred, color='orange', linestyle='--', label='Extraversion Regression Line')
plt.legend()
plt.savefig('combined_regression_plot.png')
plt.close()


# Step 5: Perform hierarchical regression analysis
y = df['Authoritative Leadership']
# Model with one indep var: extraversion
X0 = sm.add_constant(df['Extraversion'])
model0 = sm.OLS(y, X0).fit()
print("\nModel 0 Summary (Extraversion Only):")
print(model0.summary())

# Add 'Conscientiousness' as an independent variable
X1 = sm.add_constant(df[['Extraversion', 'Conscientiousness']])
model1 = sm.OLS(y, X1).fit()
print("\nModel 1 Summary (Extraversion and Conscientiousness):")
print(model1.summary())

# Add 'Openness' as an independent variable
X2 = sm.add_constant(df[['Extraversion', 'Conscientiousness', 'Openness']])
model2 = sm.OLS(y, X2).fit()
print("\nModel 2 Summary (Extraversion, Conscientiousness, and Openness):")
print(model2.summary())

# Add 'Neuroticism' as an independent variable
X3 = sm.add_constant(df[['Extraversion', 'Conscientiousness', 'Openness', 'Neuroticism']])
model3 = sm.OLS(y, X3).fit()
print("\nModel 3 Summary (Extraversion, Conscientiousness, Openness, and Neuroticism):")
print(model3.summary())

# Add 'Agreeableness' as an independent variable
X4 = sm.add_constant(df[['Extraversion', 'Conscientiousness', 'Openness', 'Neuroticism', 'Agreeableness']])
model4 = sm.OLS(y, X4).fit()
print("\nModel 4 Summary (Extraversion, Conscientiousness, Openness, Neuroticism, and Agreeableness):")
print(model4.summary())


# Hierarchical scatterplots?


# Write to csv
correlation_matrix_all.to_csv('combined_correlation_matrices.csv', mode='w', header=True)
with open('combined_correlation_matrices.csv', 'a') as file:
    file.write('\n\n')  # Add some new lines to make it look nice
correlation_matrix_personality.to_csv('combined_correlation_matrices.csv', mode='a', header=True)
with open('combined_correlation_matrices.csv', 'a') as file:
    file.write('\n\n')
personality_leadership_correlations.to_csv('combined_correlation_matrices.csv', mode='a', header=True)
with open('combined_correlation_matrices.csv', 'a') as file:
    file.write('\n\n')
# Writing multiple regression summary
model_summary = model.summary()
with open('combined_correlation_matrices.csv', 'a') as file:
    file.write(model_summary.as_csv())
    file.write('\n\n\n\n')
# Writing hierchial regression
model_summaries = [model0, model1, model2, model3, model4]
with open('combined_correlation_matrices.csv', 'a') as file:
    for model in model_summaries:
        model_summary = model.summary()
        file.write(model_summary.as_csv())
        file.write('\n\n')

