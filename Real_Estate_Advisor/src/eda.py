from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px

from src.config import EDA_REPORT_PATH, FIGURE_DIR, RAW_DATA_PATH
from src.features import engineer_features, load_raw_data
from src.utils import ensure_dir


def _save_chart(fig, name: str) -> str:
    ensure_dir(FIGURE_DIR)
    path = FIGURE_DIR / f"{name}.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return str(path)


def _money(value: float) -> str:
    return f"{value:.2f} lakhs"


def generate_eda_report() -> Path:
    df = engineer_features(load_raw_data(RAW_DATA_PATH))
    ensure_dir(EDA_REPORT_PATH.parent)

    top_city_price = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False)
    top_state_pps = df.groupby("State")["Price_per_SqFt"].mean().sort_values(ascending=False)
    top_locality = (
        df.groupby("Locality")["Price_per_SqFt"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )
    bhk_city = (
        df.groupby(["City", "BHK"]).size().reset_index(name="records").sort_values("records", ascending=False)
    )
    corr = df[
        [
            "Price_in_Lakhs",
            "Size_in_SqFt",
            "Price_per_SqFt",
            "Age_of_Property",
            "Nearby_Schools",
            "Nearby_Hospitals",
            "Investment_Score",
            "Estimated_Price_After_5Y",
        ]
    ].corr(numeric_only=True)

    chart_paths = [
        _save_chart(
            px.histogram(df, x="Price_in_Lakhs", nbins=60, title="Distribution of property prices"),
            "price_distribution",
        ),
        _save_chart(
            px.histogram(df, x="Size_in_SqFt", nbins=60, title="Distribution of property sizes"),
            "size_distribution",
        ),
        _save_chart(
            px.box(df, x="Property_Type", y="Price_per_SqFt", title="Price per sqft by property type"),
            "price_per_sqft_by_type",
        ),
        _save_chart(
            px.scatter(
                df.sample(min(6000, len(df)), random_state=11),
                x="Size_in_SqFt",
                y="Price_in_Lakhs",
                color="Good_Investment",
                title="Property size and price relationship",
            ),
            "size_price_scatter",
        ),
        _save_chart(
            px.imshow(corr, text_auto=".2f", title="Numeric feature correlation"),
            "correlation_heatmap",
        ),
    ]

    lines = [
        "# Exploratory Data Analysis Report",
        "",
        "## Dataset Health",
        f"- Rows: {len(df):,}",
        f"- Columns after engineering: {len(df.columns):,}",
        f"- Duplicate IDs: {int(df['ID'].duplicated().sum()):,}",
        f"- Missing values after cleaning: {int(df.isna().sum().sum()):,}",
        "",
        "## 1-5: Price and Size Analysis",
        f"1. Property prices range from {_money(df['Price_in_Lakhs'].min())} to {_money(df['Price_in_Lakhs'].max())}, with a median of {_money(df['Price_in_Lakhs'].median())}.",
        f"2. Property sizes range from {df['Size_in_SqFt'].min():,.0f} sqft to {df['Size_in_SqFt'].max():,.0f} sqft, with a median of {df['Size_in_SqFt'].median():,.0f} sqft.",
        f"3. The highest median price per sqft property type is {df.groupby('Property_Type')['Price_per_SqFt'].median().idxmax()}.",
        f"4. Size and price correlation is {df['Size_in_SqFt'].corr(df['Price_in_Lakhs']):.3f}, so size alone is not enough for investment judgement.",
        f"5. Price per sqft upper outliers begin around {df['Price_per_SqFt'].quantile(0.99):.3f} lakhs per sqft.",
        "",
        "## 6-10: Location Based Analysis",
        f"6. Highest average price per sqft state: {top_state_pps.index[0]} at {top_state_pps.iloc[0]:.3f} lakhs per sqft.",
        f"7. Highest average property price city: {top_city_price.index[0]} at {_money(top_city_price.iloc[0])}.",
        f"8. Median property age by locality ranges from {df.groupby('Locality')['Age_of_Property'].median().min():.0f} to {df.groupby('Locality')['Age_of_Property'].median().max():.0f} years.",
        f"9. The most common city-BHK combination is {bhk_city.iloc[0]['City']} with {int(bhk_city.iloc[0]['BHK'])} BHK properties.",
        "10. Top five expensive localities by price per sqft: "
        + ", ".join([f"{idx} ({value:.3f})" for idx, value in top_locality.items()])
        + ".",
        "",
        "## 11-15: Feature Relationships",
        f"11. Current price and five year estimate correlation is {corr.loc['Price_in_Lakhs', 'Estimated_Price_After_5Y']:.3f}.",
        f"12. Nearby schools and price per sqft correlation is {df['Nearby_Schools'].corr(df['Price_per_SqFt']):.3f}.",
        f"13. Nearby hospitals and price per sqft correlation is {df['Nearby_Hospitals'].corr(df['Price_per_SqFt']):.3f}.",
        f"14. Highest median price furnished status: {df.groupby('Furnished_Status')['Price_in_Lakhs'].median().idxmax()}.",
        f"15. Highest median price per sqft facing direction: {df.groupby('Facing')['Price_per_SqFt'].median().idxmax()}.",
        "",
        "## 16-20: Investment, Amenities, Ownership",
        f"16. Most common owner type: {df['Owner_Type'].value_counts().idxmax()}.",
        f"17. Most common availability status: {df['Availability_Status'].value_counts().idxmax()}.",
        f"18. Parking properties have median price {_money(df[df['Parking_Space'] == 'Yes']['Price_in_Lakhs'].median())}; non-parking properties have median price {_money(df[df['Parking_Space'] == 'No']['Price_in_Lakhs'].median())}.",
        f"19. Properties with at least four amenities have median price per sqft {df[df['amenity_count'] >= 4]['Price_per_SqFt'].median():.3f}.",
        f"20. High transport locations average {df[df['Public_Transport_Accessibility'] == 'High']['Investment_Score'].mean():.2f} investment score versus {df[df['Public_Transport_Accessibility'] == 'Low']['Investment_Score'].mean():.2f} for low access.",
        "",
        "## Chart Artifacts",
    ]
    lines.extend([f"- {path}" for path in chart_paths])
    lines.append("")

    EDA_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return EDA_REPORT_PATH


if __name__ == "__main__":
    output = generate_eda_report()
    print(f"EDA report written to {output}")
