# MOF-Quartile-driven synthetic sample generation 


## Overview

This project contains the **MSSG (MOF-based Synthetic Sample Generation)** class, a custom oversampling technique designed to handle imbalanced datasets. It combines **MOF (Mass-ratio-variance Outlier Factor)** scores with a hybrid approach of **Directional Oversampling** and **SMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic minority class samples.

The algorithm stratifies minority samples based on their MOF scores and applies different generation strategies accordingly to improve the quality of the synthetic data.

## Features

- **MOF Score Calculation**: Uses `pymof` to calculate scores for data instances.
- **Quartile-based Stratification**: Classifies samples into groups based on MOF score quartiles.
- **Directional Sampling**: A custom method to generate synthetic samples along the direction of nearest minority neighbors.
- **Hybrid Strategy**: Dynamically chooses between Directional Sampling and SMOTE based on the density and distribution of the minority samples in the stratified groups.

## Dependencies

To use this code, you need the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `scipy`
- `pymof`
- `smote_variants`
- `imbalanced-learn` (`imblearn`)

You can install the necessary dependencies using pip:

```bash
pip install pandas numpy matplotlib scikit-learn scipy pymof smote-variants imbalanced-learn
```

## Usage

1.  **Import the Class**: Ensure the `MSSG-SMOTE-Class.py` file is in your working directory or python path.
2.  **Prepare Data**: Your data should be separated into features (`X`) and target labels (`y`).
3.  **Instantiate and Sample**:

```python
# Assuming you actally rename the file to a standard python module name or import it directly
from MSSG-SMOTE-Class import MSSG
import pandas as pd
import numpy as np

# Example Data Preparation
# X = ... (numpy array or dataframe)
# y = ... (numpy array or series)

mssg = MSSG()
X_resampled, y_resampled = mssg.sample(X, y)

print(f"Original shape: {X.shape}")
print(f"Resampled shape: {X_resampled.shape}")
```

## Class Methods

### `mof_scores(data)`

Calculates the MOF scores for the input data using the `pymof` library.

### `add_class(data)`

Segments the data into quartiles based on their MOF scores. Returns the quartile class (1-4) for each instance.

### `Directional(data, X_maj, X_min, all_sample)`

Generates synthetic samples directionally. It finds the nearest neighbors in the majority and minority classes and generates new samples along the vector connecting the input data to its neighbors.

### `sample(X, y)`

The main method to perform the oversampling.

1.  Splits data into majority and minority classes.
2.  Identifies "safe" and "unsafe" samples based on MOF quartiles.
3.  Iterates through stratified combinations of minority and global data scores.
4.  Applies **Directional Sampling** or **SMOTE** based on the specific group characteristics and sample counts.
5.  Returns the concatenated dataset of original majority samples, "safe" minority samples, and newly generated synthetic samples.

## Notes

- The class prints detailed logs to the console during execution (e.g., number of dropped instances, ratios used, number of synthetic samples generated) to help track the process.
- The `Directional` method uses a logic involving `alpha` randomization to create samples between points.
