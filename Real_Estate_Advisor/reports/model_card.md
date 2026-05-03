# Real Estate Advisor Model Card

## Purpose
Predict whether a property is a good investment and estimate its price after five years.

## Target Design
The source data does not include historical resale outcomes. The project creates domain targets from value, infrastructure, amenities, transport access, property age, and construction status.

## Model
NumPy ridge ensemble for classification, score prediction, and 5 year price regression

## Evaluation
- Classification accuracy: 0.9437
- Precision: 0.8708
- Recall: 0.9977
- F1 score: 0.9299
- Regression RMSE: 8.2840 lakhs
- Regression MAE: 6.7263 lakhs
- Regression R2: 0.9983

## Responsible Use
The advisor is a decision support tool. It should be combined with legal diligence, site inspection, loan terms, and local market knowledge before investing.
