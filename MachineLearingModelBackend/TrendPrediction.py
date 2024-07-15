# Install necessary libraries
# !pip install fasteda
# !pip install datacleaner

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabulate import tabulate
from fasteda import fast_eda
from datacleaner import autoclean

# Suppress warnings
warnings.filterwarnings("ignore")
# Load data
df = pd.read_csv("/content/Sales_Product_Details.csv")

# Initial data inspection
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())

# Data cleaning
df = autoclean(df)
print(df.head())

# Perform EDA
fast_eda(df)

# Visualize data distribution and normality
selected_columns = [
    "Date",
    "Product_ID",
    "Quantity",
    "Product_Category",
]  # Replace with actual column names

for column in selected_columns:
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    sns.histplot(df[column], kde=True)
    plt.title(f"Histogram of {column}")

    plt.subplot(122)
    stats.probplot(df[column], dist="norm", plot=plt)
    plt.title(f"Q-Q Plot of {column}")
    plt.show()
# Data normalization
qt = QuantileTransformer(output_distribution="normal")
df[df.columns] = qt.fit_transform(df)


# Treat outliers
for col in df:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.5 * iqr
    df[col] = np.where(
        df[col] > upper_whisker,
        upper_whisker,
        np.where(df[col] < lower_whisker, lower_whisker, df[col]),
    )


# Drop unnecessary columns
df.drop("Product_Line", axis=1, inplace=True)
print(df.head())

# Extract features and target
X = df.drop("Sales_Revenue", axis=1)
Y = df["Sales_Revenue"]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Support Vector Regression": SVR(kernel="linear"),
    "Decision Tree Regression": DecisionTreeRegressor(),
    "Random Forest Regression": RandomForestRegressor(
        n_estimators=100, random_state=42
    ),
}

# Train and evaluate models
results = []

for name, model in models.items():
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    results.append(
        [
            name,
            round(mean_absolute_error(y_pred, Y_test), 2),
            round(np.sqrt(mean_squared_error(y_pred, Y_test)), 2),
            round(r2_score(y_pred, Y_test), 2),
        ]
    )

# Print results
columns = [
    "Model Name",
    "Mean Absolute Error",
    "Root Mean Squared Error",
    "R Squared Error",
]
print(tabulate(results, headers=columns, tablefmt="fancy_grid"))

# Create a DataFrame with actual and predicted values (using the best model: Random Forest Regression)
best_model = models["Random Forest Regression"]
results_df = pd.DataFrame(
    {"Actual Values": Y_test.values, "Predicted Values": best_model.predict(X_test)}
)

print(results_df.head(10))

# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, results_df["Predicted Values"], alpha=0.7, label="Predicted")
plt.plot(
    [min(Y_test), max(Y_test)],
    [min(Y_test), max(Y_test)],
    linestyle="--",
    color="red",
    label="Perfect Prediction",
)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs. Actual Values")
plt.legend()
plt.show()
