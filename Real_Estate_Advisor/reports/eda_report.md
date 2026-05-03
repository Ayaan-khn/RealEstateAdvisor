# Exploratory Data Analysis Report

## Dataset Health
- Rows: 250,000
- Columns after engineering: 47
- Duplicate IDs: 0
- Missing values after cleaning: 0

## 1-5: Price and Size Analysis
1. Property prices range from 10.00 lakhs to 500.00 lakhs, with a median of 253.87 lakhs.
2. Property sizes range from 500 sqft to 5,000 sqft, with a median of 2,747 sqft.
3. The highest median price per sqft property type is Independent House.
4. Size and price correlation is -0.003, so size alone is not enough for investment judgement.
5. Price per sqft upper outliers begin around 0.658 lakhs per sqft.

## 6-10: Location Based Analysis
6. Highest average price per sqft state: Karnataka at 0.133 lakhs per sqft.
7. Highest average property price city: Bangalore at 258.46 lakhs.
8. Median property age by locality ranges from 17 to 22 years.
9. The most common city-BHK combination is Gaya with 1 BHK properties.
10. Top five expensive localities by price per sqft: Locality_207 (0.152), Locality_416 (0.148), Locality_246 (0.146), Locality_359 (0.145), Locality_387 (0.144).

## 11-15: Feature Relationships
11. Current price and five year estimate correlation is 0.992.
12. Nearby schools and price per sqft correlation is -0.000.
13. Nearby hospitals and price per sqft correlation is -0.000.
14. Highest median price furnished status: Unfurnished.
15. Highest median price per sqft facing direction: West.

## 16-20: Investment, Amenities, Ownership
16. Most common owner type: Broker.
17. Most common availability status: Under_Construction.
18. Parking properties have median price 254.36 lakhs; non-parking properties have median price 253.37 lakhs.
19. Properties with at least four amenities have median price per sqft 0.093.
20. High transport locations average 59.34 investment score versus 51.10 for low access.

## Chart Artifacts
- C:\Users\mikez\PycharmProjects\RealEstateAdvisor\Real_Estate_Advisor\reports\figures\price_distribution.html
- C:\Users\mikez\PycharmProjects\RealEstateAdvisor\Real_Estate_Advisor\reports\figures\size_distribution.html
- C:\Users\mikez\PycharmProjects\RealEstateAdvisor\Real_Estate_Advisor\reports\figures\price_per_sqft_by_type.html
- C:\Users\mikez\PycharmProjects\RealEstateAdvisor\Real_Estate_Advisor\reports\figures\size_price_scatter.html
- C:\Users\mikez\PycharmProjects\RealEstateAdvisor\Real_Estate_Advisor\reports\figures\correlation_heatmap.html
