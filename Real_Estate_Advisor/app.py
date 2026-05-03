from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    ADVISOR_MODEL_PATH,
    CLEAN_DATA_PATH,
    RAW_DATA_PATH,
    REPORT_DIR,
)
from src.features import (
    AMENITY_KEYWORDS,
    clean_dataset,
    engineer_features,
    load_raw_data,
)
from src.modeling import RealEstateAdvisorModel
from src.train import train_pipeline


st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="REA",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    [data-testid="stMetricValue"] {font-size: 1.6rem;}
    .small-caption {color: #5f6b7a; font-size: 0.86rem;}
    .status-positive {color: #16794c; font-weight: 700;}
    .status-negative {color: #b42318; font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_market_data() -> pd.DataFrame:
    if CLEAN_DATA_PATH.exists():
        data = pd.read_csv(CLEAN_DATA_PATH)
    else:
        data = clean_dataset(load_raw_data(RAW_DATA_PATH))
    return engineer_features(data)


@st.cache_resource(show_spinner=False)
def load_model() -> RealEstateAdvisorModel:
    if not ADVISOR_MODEL_PATH.exists():
        train_pipeline()
    return RealEstateAdvisorModel.load(ADVISOR_MODEL_PATH)


def option_list(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).unique().tolist())


def default_value(options: list[str], preferred: str | None = None) -> str:
    if preferred in options:
        return preferred
    return options[0] if options else ""


def filtered_market_frame(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Market Filters")

        states = ["All"] + option_list(df["State"])
        state = st.selectbox("State", states)

        state_df = df if state == "All" else df[df["State"] == state]
        cities = ["All"] + option_list(state_df["City"])
        city = st.selectbox("City", cities)

        city_df = state_df if city == "All" else state_df[state_df["City"] == city]
        property_types = ["All"] + option_list(city_df["Property_Type"])
        property_type = st.selectbox("Property type", property_types)

        bhk_values = ["All"] + [str(int(x)) for x in sorted(city_df["BHK"].dropna().unique())]
        bhk = st.selectbox("BHK", bhk_values)

        min_price = float(df["Price_in_Lakhs"].min())
        max_price = float(df["Price_in_Lakhs"].max())
        price_range = st.slider(
            "Price range in lakhs",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price),
            step=5.0,
        )

    out = df.copy()
    if state != "All":
        out = out[out["State"] == state]
    if city != "All":
        out = out[out["City"] == city]
    if property_type != "All":
        out = out[out["Property_Type"] == property_type]
    if bhk != "All":
        out = out[out["BHK"] == int(bhk)]
    out = out[
        (out["Price_in_Lakhs"] >= price_range[0])
        & (out["Price_in_Lakhs"] <= price_range[1])
    ]
    return out


def advisor_form(df: pd.DataFrame, model: RealEstateAdvisorModel) -> pd.DataFrame:
    cities = option_list(df["City"])
    states = option_list(df["State"])
    property_types = option_list(df["Property_Type"])
    localities = option_list(df["Locality"])

    median_row = df.loc[df["Price_in_Lakhs"].sub(df["Price_in_Lakhs"].median()).abs().idxmin()]

    with st.form("advisor_form"):
        st.subheader("Property Input")

        c1, c2, c3, c4 = st.columns(4)
        state = c1.selectbox("State", states, index=states.index(default_value(states, str(median_row["State"]))))
        city = c2.selectbox("City", cities, index=cities.index(default_value(cities, str(median_row["City"]))))
        locality = c3.selectbox(
            "Locality",
            localities,
            index=localities.index(default_value(localities, str(median_row["Locality"]))),
        )
        property_type = c4.selectbox(
            "Property type",
            property_types,
            index=property_types.index(default_value(property_types, str(median_row["Property_Type"]))),
        )

        c1, c2, c3, c4 = st.columns(4)
        bhk = c1.number_input("BHK", min_value=1, max_value=8, value=int(median_row["BHK"]), step=1)
        size = c2.number_input(
            "Size in sqft",
            min_value=250,
            max_value=12000,
            value=int(median_row["Size_in_SqFt"]),
            step=50,
        )
        price = c3.number_input(
            "Current price in lakhs",
            min_value=5.0,
            max_value=2000.0,
            value=float(round(median_row["Price_in_Lakhs"], 2)),
            step=5.0,
        )
        year_built = c4.number_input(
            "Year built",
            min_value=1980,
            max_value=2026,
            value=int(median_row["Year_Built"]),
            step=1,
        )

        c1, c2, c3, c4 = st.columns(4)
        furnished = c1.selectbox("Furnished status", option_list(df["Furnished_Status"]))
        floor_no = c2.number_input("Floor no", min_value=0, max_value=80, value=int(median_row["Floor_No"]), step=1)
        total_floors = c3.number_input(
            "Total floors",
            min_value=1,
            max_value=80,
            value=max(1, int(median_row["Total_Floors"])),
            step=1,
        )
        facing = c4.selectbox("Facing", option_list(df["Facing"]))

        c1, c2, c3, c4 = st.columns(4)
        schools = c1.slider("Nearby schools", 0, 15, int(median_row["Nearby_Schools"]))
        hospitals = c2.slider("Nearby hospitals", 0, 15, int(median_row["Nearby_Hospitals"]))
        transport = c3.selectbox(
            "Public transport",
            option_list(df["Public_Transport_Accessibility"]),
        )
        availability = c4.selectbox("Availability", option_list(df["Availability_Status"]))

        c1, c2, c3 = st.columns(3)
        parking = c1.selectbox("Parking space", option_list(df["Parking_Space"]))
        security = c2.selectbox("Security", option_list(df["Security"]))
        owner_type = c3.selectbox("Owner type", option_list(df["Owner_Type"]))

        selected_amenities = st.multiselect(
            "Amenities",
            AMENITY_KEYWORDS,
            default=["Gym", "Pool", "Garden"],
        )

        submitted = st.form_submit_button("Run investment analysis", width="stretch")

    record = pd.DataFrame(
        [
            {
                "ID": 0,
                "State": state,
                "City": city,
                "Locality": locality,
                "Property_Type": property_type,
                "BHK": bhk,
                "Size_in_SqFt": size,
                "Price_in_Lakhs": price,
                "Price_per_SqFt": price / max(size, 1),
                "Year_Built": year_built,
                "Furnished_Status": furnished,
                "Floor_No": floor_no,
                "Total_Floors": total_floors,
                "Age_of_Property": 2026 - year_built,
                "Nearby_Schools": schools,
                "Nearby_Hospitals": hospitals,
                "Public_Transport_Accessibility": transport,
                "Parking_Space": parking,
                "Security": security,
                "Amenities": ", ".join(selected_amenities),
                "Facing": facing,
                "Owner_Type": owner_type,
                "Availability_Status": availability,
            }
        ]
    )

    prediction = model.predict(record)
    if submitted:
        st.session_state["last_prediction"] = prediction
    return st.session_state.get("last_prediction", prediction)


def prediction_panel(prediction: pd.DataFrame) -> None:
    row = prediction.iloc[0]
    is_good = bool(row["Good_Investment_Prediction"])
    status_class = "status-positive" if is_good else "status-negative"
    status_text = "Good investment" if is_good else "Needs caution"

    st.markdown(f"<div class='{status_class}'>{status_text}</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Investment probability", f"{row['Investment_Probability'] * 100:.1f}%")
    c2.metric("Confidence", f"{row['Model_Confidence'] * 100:.1f}%")
    c3.metric("5 year forecast", f"{row['Predicted_5Y_Price_Lakhs']:.2f} L")
    c4.metric("Expected annual growth", f"{row['Annual_Appreciation_Rate'] * 100:.2f}%")

    st.progress(float(row["Investment_Probability"]))

    drivers = row["Drivers"]
    st.subheader("Decision Drivers")
    d1, d2, d3 = st.columns(3)
    for idx, driver in enumerate(drivers[:6]):
        [d1, d2, d3][idx % 3].info(driver)


def market_tab(df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    if filtered_df.empty:
        st.warning("No properties match the selected filters.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filtered records", f"{len(filtered_df):,}")
    c2.metric("Average price", f"{filtered_df['Price_in_Lakhs'].mean():.2f} L")
    c3.metric("Median price per sqft", f"{filtered_df['Price_per_SqFt'].median():.3f} L")
    c4.metric("Good investment share", f"{filtered_df['Good_Investment'].mean() * 100:.1f}%")

    c1, c2 = st.columns(2)
    top_city = (
        df.groupby("City", as_index=False)
        .agg(
            avg_price=("Price_in_Lakhs", "mean"),
            avg_investment_score=("Investment_Score", "mean"),
            records=("ID", "count"),
        )
        .sort_values("avg_investment_score", ascending=False)
        .head(12)
    )
    fig_city = px.bar(
        top_city,
        x="City",
        y="avg_investment_score",
        color="avg_price",
        title="Top cities by investment score",
        labels={"avg_investment_score": "Investment score", "avg_price": "Avg price"},
        color_continuous_scale="Tealrose",
    )
    c1.plotly_chart(fig_city, width="stretch")

    type_data = filtered_df.groupby("Property_Type", as_index=False)["Price_per_SqFt"].median()
    fig_type = px.bar(
        type_data,
        x="Property_Type",
        y="Price_per_SqFt",
        title="Median price per sqft by property type",
        labels={"Price_per_SqFt": "Lakhs per sqft"},
        color="Property_Type",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    c2.plotly_chart(fig_type, width="stretch")

    c1, c2 = st.columns(2)
    sample = filtered_df.sample(min(4000, len(filtered_df)), random_state=7)
    fig_scatter = px.scatter(
        sample,
        x="Size_in_SqFt",
        y="Price_in_Lakhs",
        color="Good_Investment",
        hover_data=["City", "Property_Type", "BHK"],
        title="Size vs price with investment label",
        color_discrete_map={True: "#16794c", False: "#b42318"},
    )
    c1.plotly_chart(fig_scatter, width="stretch")

    trend = (
        filtered_df.groupby("Age_of_Property", as_index=False)
        .agg(avg_price=("Price_in_Lakhs", "mean"), avg_forecast=("Estimated_Price_After_5Y", "mean"))
        .sort_values("Age_of_Property")
    )
    fig_age = px.line(
        trend,
        x="Age_of_Property",
        y=["avg_price", "avg_forecast"],
        title="Current price vs 5 year estimate by property age",
        labels={"value": "Lakhs", "variable": "Metric"},
    )
    c2.plotly_chart(fig_age, width="stretch")


def model_report_tab(model: RealEstateAdvisorModel) -> None:
    st.subheader("Model Performance")

    metrics = model.metrics or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics.get('classification_accuracy', 0) * 100:.2f}%")
    c2.metric("F1 score", f"{metrics.get('classification_f1', 0) * 100:.2f}%")
    c3.metric("RMSE", f"{metrics.get('regression_rmse', 0):.2f} L")
    c4.metric("R2", f"{metrics.get('regression_r2', 0):.3f}")

    st.subheader("Feature Importance")
    importance = model.feature_importance()
    if importance.empty:
        st.info("Feature importance will be available after model training.")
    else:
        fig = px.bar(
            importance.head(20).sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Top model drivers",
            labels={"importance": "Relative weight", "feature": "Feature"},
            color="importance",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig, width="stretch")

    card_path = REPORT_DIR / "model_card.md"
    if card_path.exists():
        st.download_button(
            "Download model card",
            data=card_path.read_text(encoding="utf-8"),
            file_name="real_estate_model_card.md",
            mime="text/markdown",
            width="stretch",
        )


def data_tab(df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    c1, c2 = st.columns([1, 2])
    c1.metric("Raw rows", f"{len(df):,}")
    c1.metric("Cities", f"{df['City'].nunique():,}")
    c1.metric("Localities", f"{df['Locality'].nunique():,}")
    c2.dataframe(filtered_df.head(1000), width="stretch", height=420)


def main() -> None:
    with st.spinner("Loading advisor assets"):
        df = load_market_data()
        model = load_model()

    st.title("Real Estate Investment Advisor")
    st.caption("Property profitability classification, five year value forecasting, and market intelligence.")

    filtered_df = filtered_market_frame(df)
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Advisor", "Market Intelligence", "Model Report", "Data Explorer"]
    )

    with tab1:
        prediction = advisor_form(df, model)
        prediction_panel(prediction)

    with tab2:
        market_tab(df, filtered_df)

    with tab3:
        model_report_tab(model)

    with tab4:
        data_tab(df, filtered_df)


if __name__ == "__main__":
    main()
