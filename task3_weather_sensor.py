import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, zscore
# 1. Creating the weather sensor dataset
data = {
    "Hour": range(1, 16),
    "Temperature": [
        22, 23, np.nan, np.nan, np.nan,
        25, 26, 58, 24, 23,
        22, 21, 23, 24, 25
    ]
}
df = pd.DataFrame(data)
print("Original Temperature Data:")
print(df)
# 2. Descriptive statistics BEFORE cleaning
print("\nStats BEFORE cleaning:")
print("Mean:", df["Temperature"].mean())
print("Standard Deviation:", df["Temperature"].std())
# Remove missing values before calculating skewness and kurtosis
before_clean = df["Temperature"].dropna()
print("Skewness:", skew(before_clean))
print("Kurtosis:", kurtosis(before_clean))
# 3. Fill missing values using linear interpolation
df["Temperature_Clean"] = df["Temperature"].interpolate(method="linear")
print("\nAfter interpolation, missing values:",
      df["Temperature_Clean"].isnull().sum())
# 4. Detect outlier using Z-score threshold = 2
z_scores = zscore(df["Temperature_Clean"])
df["Zscore"] = z_scores
outlier_mask = df["Zscore"].abs() > 2
print("\nZ-score(>2) outlier detected:")
print(df.loc[outlier_mask, ["Hour", "Temperature_Clean", "Zscore"]])
# 5. Calculate median
median_temperature = df["Temperature_Clean"].median()
print("\nMedian Temperature:", median_temperature)
# 6. Replace ONLY the flagged outlier with median
df.loc[outlier_mask, "Temperature_Clean"] = median_temperature
print("\nAfter replacing outlier:")
print(df[["Hour", "Temperature", "Temperature_Clean"]])
# 7. Standardize the final cleaned temperature
df["Temperature_Zscore"] = zscore(df["Temperature_Clean"])
# 8. Before/After comparison table
comparison = df[
    ["Hour", "Temperature", "Temperature_Clean", "Temperature_Zscore"]
]
print("\nBefore/After Comparison:")
print(comparison)
# 9. Statistics AFTER cleaning
print("\nStats AFTER cleaning:")
print("Mean:", df["Temperature_Clean"].mean())
print("Standard Deviation:", df["Temperature_Clean"].std())