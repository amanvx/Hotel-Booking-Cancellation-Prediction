# Hotel Booking Cancellation Prediction

Personal machine learning project that predicts whether a hotel reservation will be cancelled, using features available at booking time. Built with Python, Pandas, and Scikit-learn.

## Business Problem

Hotels lose revenue when guests cancel bookings late or don't show up — rooms held for cancelled reservations can't be resold, and last-minute cancellations leave little time to fill the gap. Predicting cancellation risk at (or shortly after) booking time lets a hotel:

- Apply targeted overbooking strategies for high-risk segments
- Trigger confirmation calls or incentives for bookings flagged as high risk
- Adjust pricing or deposit policies for risky market segments / lead times
- Improve revenue forecasting accuracy

This is framed as a **binary classification problem**: predict booking status (Cancelled / Not Cancelled) from reservation attributes.

## Dataset

Kaggle ["Hotel Reservations" dataset](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset) — ~36,000 bookings, 17 columns, including guest composition, stay length, meal plan, room type, lead time, market segment, prior cancellation history, price, and special requests.

## Approach

1. **Preprocessing** — dropped the booking ID, engineered calendar features (month, day-of-week) from the reservation date, encoded categorical columns, and checked for missing values / leakage-prone features.
2. **Exploratory Data Analysis** — examined class balance, lead time vs. cancellation, cancellation rate by market segment, and correlations between numeric features.
3. **Feature Engineering** — created `total_nights`, `total_guests`, and `price_per_night`; label-encoded categoricals.
4. **Modeling** — trained and compared three models spanning different bias/variance trade-offs:
   - **Logistic Regression** — interpretable linear baseline
   - **Decision Tree** — captures non-linear splits, useful as an interpretable reference
   - **Random Forest** — ensemble model, generally the strongest baseline on this kind of tabular data
5. **Hyperparameter Tuning** — grid search with stratified 5-fold cross-validation on the Random Forest, optimizing for F1.
6. **Evaluation** — compared models using F1 and AUC (rather than raw accuracy) due to class imbalance, and examined feature importances for business interpretability.

## Why F1/AUC Over Accuracy

The classes are imbalanced, so accuracy alone rewards the majority class. Business cost is also asymmetric: missing a likely cancellation (false negative) wastes a room, while flagging a booking that wasn't going to cancel (false positive) at worst triggers an unnecessary confirmation email. This makes recall on the "Cancelled" class a business-relevant metric alongside F1 and AUC.

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (Logistic Regression, Decision Tree, Random Forest, GridSearchCV)
- Matplotlib, Seaborn

## Repository Contents

- `notebook.ipynb` — full analysis: EDA, feature engineering, model training, tuning, and evaluation

## Future Improvements

- Extend model comparison to gradient boosting (LightGBM / XGBoost) and a stacking ensemble
- Address class imbalance explicitly with `class_weight="balanced"`, SMOTE, or threshold tuning
- Add SHAP values for per-booking explainability
- Deploy as a lightweight scoring API integrated with a hotel's reservation system
- Incorporate external features such as local events, weather, or seasonality indices

## Author

Built as a personal project to demonstrate an end-to-end ML workflow: problem framing, EDA, feature engineering, model comparison, hyperparameter tuning, and business-oriented evaluation.
