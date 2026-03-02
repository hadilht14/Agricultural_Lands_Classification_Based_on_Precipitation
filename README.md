#  Agricultural Lands Classification Based on Precipitation

**Data Mining Project**

## Project Overview

This project focuses on classifying agricultural lands in **Algeria** based on monthly precipitation data.
Using Earth observation data, we applied data mining techniques to preprocess, clean, model, and evaluate classification algorithms to determine agricultural land patterns according to rainfall distribution.

The project integrates spatial data processing, interpolation techniques, outlier detection, and machine learning optimization.

---

## Data Collection

* Data source: Earth Data (Giovanni platform – NASA Earth observation system)
* Time period:

  * **2023** → Training dataset
  * **2022** → Testing dataset
* Format:

  * Originally in **NetCDF (.nc)** format
  * Converted to **CSV** for preprocessing and modeling
* Each monthly file:

  * ~38,248 instances
  * Attributes:

    * Longitude
    * Latitude
    * Monthly average precipitation

---

## Data Cleaning & Preprocessing

Each `month_year` file was processed independently to avoid mixing precipitation patterns across months.

###  Missing Values Handling

* ~40% missing values per file
* Missing values represented as: `-9999.9`
* Since covering the full Algerian territory was essential, removing rows was not an option.

**Solution:**

* Applied **Nearest Neighbors interpolation**
* Used:

  * `scipy.interpolate.griddata`
* Each missing value was filled using the nearest geographical precipitation values based on longitude and latitude.

---

### Outlier Detection

Due to geographical variation (north vs south Algeria), standard global outlier detection would be incorrect.

**Approach:**

* Algeria was divided into circles with radius **0.5**
* Total circles: **837**
* Used:

  * `scipy.spatial.CKDTree` for spatial grouping
* Within each circle:

  * Applied **IQR (Interquartile Range)** method
  * Flagged outliers using `isOutlier = True/False`

Statistics:

* ~409 circles per file contained outliers
* ~7% of total data were outliers (on average)

**Handling:**

* Replaced outliers using the same Nearest Neighbor interpolation approach
* Verified corrections using statistical summaries

---

## Feature Engineering

* Combined all **2023** monthly data → Training dataset
* Combined all **2022** monthly data → Testing dataset
* Added new features:

  * `season` (Winter, Spring, Summer, Autumn)
  * `month_year`
  * Cyclical month encoding:

    * `sin(month)`
    * `cos(month)`
  * (To better capture seasonal rainfall patterns)

---

##  Modeling & Evaluation

The following machine learning models were applied:

* K-Nearest Neighbors (KNN)
* Random Forest (RF)
* Support Vector Machine (SVM)

Both baseline and optimized versions were evaluated.

---

##  Best Performing Models

### Optimized Random Forest

Accuracy: **100%**

Optimization techniques:

* Strategic hyperparameter tuning
* Cyclical feature engineering (sin/cos for months)

---

###  Optimized K-Nearest Neighbors

Accuracy: **100%**

Optimization techniques:

* RandomizedSearchCV for hyperparameter tuning

---

###  Optimized Support Vector Machine

Accuracy: **94%**

Optimization techniques:

* Cross-validation
* PCA (Principal Component Analysis)

---

### Baseline Models Performance

| Model                  | Accuracy |
| ---------------------- | -------- |
| Random Forest          | 91%      |
| Support Vector Machine | 90%      |

---

##  Technologies & Libraries Used

* Python
* NumPy
* Pandas
* Scikit-learn
* SciPy
* NetCDF4
* Matplotlib / Seaborn (for visualization)

---

## Key Contributions

✔ Large-scale spatial data preprocessing
✔ Advanced missing value handling using spatial interpolation
✔ Localized outlier detection using spatial clustering
✔ Feature engineering for seasonal pattern recognition
✔ Hyperparameter optimization for improved performance

---

## Conclusion

This project demonstrates how spatial data mining and machine learning can effectively classify agricultural lands based on precipitation patterns.

By combining geospatial processing, statistical techniques, and optimized ML models, we achieved highly accurate classification results.

