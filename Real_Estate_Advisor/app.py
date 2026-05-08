import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from predict import predict


st.set_page_config(
    page_title="Real Estate Advisor",
    page_icon="REA",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_property_data() -> pd.DataFrame:
    data = pd.read_csv("data/raw/india_housing_prices.csv")
    data["Amenities_List"] = (
        data["Amenities"].fillna("").str.split(",").apply(lambda values: [v.strip() for v in values if v.strip()])
    )
    data["Amenity_Count"] = data["Amenities_List"].apply(len)
    data["Property_Title"] = (
        data["BHK"].astype(str)
        + " BHK "
        + data["Property_Type"].astype(str)
        + " in "
        + data["Locality"].astype(str)
    )
    data["Estimated_Rent_Lakhs"] = (data["Price_in_Lakhs"] * 0.0028).round(2)
    data["Annual_Rent_Lakhs"] = (data["Estimated_Rent_Lakhs"] * 12).round(2)
    data["Rental_Yield"] = ((data["Annual_Rent_Lakhs"] / data["Price_in_Lakhs"]) * 100).round(2)
    data["Maintenance_Lakhs"] = (data["Price_in_Lakhs"] * 0.0007).round(2)
    data["Net_Cashflow_Lakhs"] = (
        data["Estimated_Rent_Lakhs"] - data["Maintenance_Lakhs"]
    ).round(2)
    data["Infra_Score"] = (
        data["Nearby_Schools"].fillna(0) * 1.15
        + data["Nearby_Hospitals"].fillna(0)
        + data["Amenity_Count"] * 0.55
        + data["Public_Transport_Accessibility"].map({"Low": 1, "Medium": 2, "High": 3}).fillna(1) * 1.4
    ).round(1)
    data["Price_per_SqFt_Display"] = (
        (data["Price_in_Lakhs"] * 100000) / data["Size_in_SqFt"]
    ).replace([np.inf, -np.inf], 0).fillna(0).round(0).astype(int)
    return data


df = load_property_data()


st.markdown(
    """
    <style>
        :root {
            --teal: #08a8a6;
            --green: #19b36b;
            --blue: #4f75f6;
            --ink: #202631;
            --muted: #697384;
            --line: #e9edf3;
            --surface: #f5f8fb;
        }

        .stApp {
            background: linear-gradient(180deg, #eef5fa 0, #f8fbfd 260px, #ffffff 100%);
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--ink);
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px 18px 14px;
            box-shadow: 0 12px 35px rgba(33, 44, 69, 0.06);
        }

        div[data-testid="stMetric"] label {
            color: var(--muted);
        }

        .panel {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 12px 35px rgba(33, 44, 69, 0.06);
        }

        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--ink);
            margin: 0 0 0.2rem;
        }

        .section-note {
            color: var(--muted);
            font-size: 0.88rem;
            margin-bottom: 0.75rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 0.78rem;
            font-weight: 700;
            background: #edf8f6;
            color: #067c7a;
            border: 1px solid #d6efec;
        }

        .hero {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 22px;
            box-shadow: 0 16px 42px rgba(33, 44, 69, 0.07);
        }

        .hero h1 {
            font-size: clamp(1.6rem, 3vw, 2.45rem);
            line-height: 1.05;
            margin: 0.35rem 0 0.55rem;
        }

        .hero p {
            color: var(--muted);
            max-width: 760px;
            margin: 0;
            font-size: 1rem;
        }

        .status-good {
            color: #087f4f;
            background: #eaf8f0;
            border-color: #ccebd9;
        }

        .status-watch {
            color: #8a5b00;
            background: #fff7df;
            border-color: #f5df9b;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: #ffffff;
            border-radius: 999px;
            border: 1px solid var(--line);
            padding: 10px 18px;
        }

        .stTabs [aria-selected="true"] {
            background: #202631;
            color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_lakhs(value: float) -> str:
    if pd.isna(value):
        return "0 L"
    if value >= 100:
        return f"{value / 100:.2f} Cr"
    return f"{value:.1f} L"


def monthly_emi(principal_lakhs: float, annual_rate: float, years: int) -> float:
    principal = principal_lakhs * 100000
    monthly_rate = annual_rate / 12 / 100
    months = max(years * 12, 1)
    if monthly_rate == 0:
        return principal / months / 100000
    emi = principal * monthly_rate * ((1 + monthly_rate) ** months) / (((1 + monthly_rate) ** months) - 1)
    return emi / 100000


def amenity_match(row: pd.Series, selected: list[str]) -> bool:
    if not selected:
        return True
    available = set(row["Amenities_List"])
    return all(item in available for item in selected)


with st.sidebar:
    st.markdown("### Real Estate Advisor")
    st.caption("Search, compare, analyze, and shortlist investment-ready homes.")

    states = sorted(df["State"].dropna().unique())
    state = st.selectbox("State", states, index=0)

    state_df = df[df["State"] == state]
    cities = sorted(state_df["City"].dropna().unique())
    selected_cities = st.multiselect("City", cities, default=cities[:1])
    city_df = state_df[state_df["City"].isin(selected_cities)] if selected_cities else state_df

    localities = sorted(city_df["Locality"].dropna().unique())
    selected_localities = st.multiselect("Locality", localities, default=[])

    landmark_search = st.text_input("Landmark search", placeholder="Type locality, school, metro, market")

    st.divider()
    st.markdown("### Property Filters")

    min_price = int(math.floor(df["Price_in_Lakhs"].min()))
    max_price = int(math.ceil(df["Price_in_Lakhs"].max()))
    default_upper = min(max_price, max(min_price + 50, 350))
    price_range = st.slider("Price range (Lakhs)", min_price, max_price, (min_price, default_upper), step=10)

    min_area = int(math.floor(df["Size_in_SqFt"].min()))
    max_area = int(math.ceil(df["Size_in_SqFt"].max()))
    area_range = st.slider("Area range (SqFt)", min_area, max_area, (min_area, max_area), step=100)

    bhk_values = sorted(df["BHK"].dropna().astype(int).unique().tolist())
    bhk_range = st.slider("BHK", min(bhk_values), max(bhk_values), (min(bhk_values), min(4, max(bhk_values))))

    property_types = sorted(df["Property_Type"].dropna().unique())
    selected_types = st.multiselect("Property type", property_types, default=property_types)

    furnishing_values = sorted(df["Furnished_Status"].dropna().unique())
    selected_furnishing = st.multiselect("Furnishing", furnishing_values, default=furnishing_values)

    st.divider()
    st.markdown("### Amenities And Nearby")

    all_amenities = sorted({item for values in df["Amenities_List"] for item in values})
    selected_amenities = st.multiselect("Amenities", all_amenities, default=[])
    min_schools = st.slider("Minimum nearby schools", 0, int(df["Nearby_Schools"].max()), 0)
    min_hospitals = st.slider("Minimum nearby hospitals", 0, int(df["Nearby_Hospitals"].max()), 0)
    transport = st.radio(
        "Transport access",
        options=["Any", "Low", "Medium", "High"],
        horizontal=True,
        index=0,
    )
    parking_required = st.toggle("Parking required", value=False)

    st.divider()
    st.markdown("### Cashflow Assumptions")
    down_payment_pct = st.slider("Down payment (%)", 10, 60, 20)
    loan_rate = st.slider("Loan rate (%)", 6.0, 12.5, 8.5, step=0.1)
    loan_years = st.slider("Loan tenure (years)", 5, 30, 20)


filtered_df = df.copy()
filtered_df = filtered_df[filtered_df["State"] == state]
if selected_cities:
    filtered_df = filtered_df[filtered_df["City"].isin(selected_cities)]
if selected_localities:
    filtered_df = filtered_df[filtered_df["Locality"].isin(selected_localities)]
if landmark_search:
    search = landmark_search.strip().lower()
    filtered_df = filtered_df[
        filtered_df["Locality"].str.lower().str.contains(search, na=False)
        | filtered_df["City"].str.lower().str.contains(search, na=False)
        | filtered_df["Amenities"].str.lower().str.contains(search, na=False)
    ]

filtered_df = filtered_df[
    filtered_df["Price_in_Lakhs"].between(price_range[0], price_range[1])
    & filtered_df["Size_in_SqFt"].between(area_range[0], area_range[1])
    & filtered_df["BHK"].between(bhk_range[0], bhk_range[1])
    & filtered_df["Property_Type"].isin(selected_types)
    & filtered_df["Furnished_Status"].isin(selected_furnishing)
    & (filtered_df["Nearby_Schools"] >= min_schools)
    & (filtered_df["Nearby_Hospitals"] >= min_hospitals)
]

if selected_amenities:
    filtered_df = filtered_df[filtered_df.apply(lambda row: amenity_match(row, selected_amenities), axis=1)]
if transport != "Any":
    filtered_df = filtered_df[filtered_df["Public_Transport_Accessibility"] == transport]
if parking_required:
    filtered_df = filtered_df[filtered_df["Parking_Space"] == "Yes"]

filtered_df = filtered_df.sort_values(
    by=["Infra_Score", "Rental_Yield", "Price_in_Lakhs"],
    ascending=[False, False, True],
)


st.markdown(
    """
    <div class="hero">
        <span class="badge">Investment dashboard</span>
        <h1>Find property opportunities with price, cashflow, and location signals in one place.</h1>
        <p>Filter by state, city, locality, landmark, price, BHK, amenities, nearby schools, hospitals, parking, and transport access. Select a listing to run investment analysis instantly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

total_inventory = len(filtered_df)
avg_price = filtered_df["Price_in_Lakhs"].mean() if total_inventory else 0
avg_yield = filtered_df["Rental_Yield"].mean() if total_inventory else 0
avg_cashflow = filtered_df["Net_Cashflow_Lakhs"].mean() if total_inventory else 0
high_infra = int((filtered_df["Infra_Score"] >= filtered_df["Infra_Score"].quantile(0.75)).sum()) if total_inventory else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Matching Properties", f"{total_inventory:,}", "Filtered inventory")
kpi2.metric("Average Price", format_lakhs(avg_price), "Asking value")
kpi3.metric("Average Rental Yield", f"{avg_yield:.2f}%", "Estimated annual")
kpi4.metric("High Infra Matches", f"{high_infra:,}", "Top-location signals")

st.write("")

tab_dashboard, tab_listings, tab_cashflow, tab_market = st.tabs(
    ["Dashboard", "Property Listings", "Cashflow Reports", "Market Analysis"]
)

with tab_dashboard:
    top_left, top_right = st.columns([1.45, 1], gap="large")

    with top_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Price And Inventory Overview</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-note">Distribution of filtered homes by budget and configuration.</p>',
            unsafe_allow_html=True,
        )
        if filtered_df.empty:
            st.info("No properties match the current filters. Loosen one filter to see results.")
        else:
            fig = px.histogram(
                filtered_df,
                x="Price_in_Lakhs",
                color="Property_Type",
                nbins=24,
                labels={"Price_in_Lakhs": "Price in Lakhs", "count": "Properties"},
                color_discrete_sequence=["#08a8a6", "#4f75f6", "#19b36b", "#f0b429"],
            )
            fig.update_layout(
                height=345,
                margin=dict(l=5, r=5, t=5, b=5),
                legend_title_text="",
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
            )
            st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with top_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Best Match To Analyze</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-note">The first listing is ranked by infrastructure, yield, and value.</p>',
            unsafe_allow_html=True,
        )
        if filtered_df.empty:
            st.warning("No selected property available.")
        else:
            selected_row = filtered_df.iloc[0]
            status_class = "status-good" if selected_row["Net_Cashflow_Lakhs"] > 0 else "status-watch"
            st.markdown(
                f"""
                <span class="badge {status_class}">{selected_row["Availability_Status"].replace("_", " ")}</span>
                <h3 style="margin: 0.65rem 0 0.3rem;">{selected_row["Property_Title"]}</h3>
                <p style="color:#697384;margin:0 0 1rem;">{selected_row["City"]}, {selected_row["State"]}</p>
                """,
                unsafe_allow_html=True,
            )
            p1, p2 = st.columns(2)
            p1.metric("Price", format_lakhs(selected_row["Price_in_Lakhs"]))
            p2.metric("Area", f"{int(selected_row['Size_in_SqFt']):,} SqFt")
            p3, p4 = st.columns(2)
            p3.metric("Yield", f"{selected_row['Rental_Yield']:.2f}%")
            p4.metric("Infra Score", f"{selected_row['Infra_Score']:.1f}")
            st.caption(
                f"Nearby: {int(selected_row['Nearby_Schools'])} schools, "
                f"{int(selected_row['Nearby_Hospitals'])} hospitals, "
                f"{selected_row['Public_Transport_Accessibility'].lower()} transport."
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    analysis_left, analysis_right = st.columns([1.1, 1], gap="large")

    with analysis_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Selected Property Price Analysis</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-note">Choose any filtered property and adjust price, BHK, and nearby signals.</p>',
            unsafe_allow_html=True,
        )
        if filtered_df.empty:
            st.info("Use the filters to bring properties into the analysis panel.")
        else:
            listing_options = filtered_df["Property_Title"] + " | " + filtered_df["City"] + " | ID " + filtered_df["ID"].astype(str)
            selected_label = st.selectbox("Select property", listing_options.tolist(), label_visibility="collapsed")
            selected_id = int(selected_label.split("ID ")[-1])
            base_row = filtered_df[filtered_df["ID"] == selected_id].iloc[0]

            c1, c2, c3 = st.columns(3)
            analysis_price = c1.slider(
                "Analysis price (Lakhs)",
                min_price,
                max_price,
                int(round(base_row["Price_in_Lakhs"])),
                step=5,
            )
            analysis_bhk = c2.slider(
                "BHK for analysis",
                min(bhk_values),
                max(bhk_values),
                int(base_row["BHK"]),
            )
            analysis_size = c3.number_input(
                "Size (SqFt)",
                min_value=min_area,
                max_value=max_area,
                value=int(base_row["Size_in_SqFt"]),
                step=50,
            )

            c4, c5, c6 = st.columns(3)
            analysis_schools = c4.slider("Schools nearby", 0, int(df["Nearby_Schools"].max()), int(base_row["Nearby_Schools"]))
            analysis_hospitals = c5.slider("Hospitals nearby", 0, int(df["Nearby_Hospitals"].max()), int(base_row["Nearby_Hospitals"]))
            analysis_transport = c6.selectbox(
                "Transport",
                ["Low", "Medium", "High"],
                index=["Low", "Medium", "High"].index(base_row["Public_Transport_Accessibility"]),
            )

            input_data = {
                "State": base_row["State"],
                "City": base_row["City"],
                "Locality": base_row["Locality"],
                "Property_Type": base_row["Property_Type"],
                "BHK": analysis_bhk,
                "Size_in_SqFt": analysis_size,
                "Price_in_Lakhs": analysis_price,
                "Year_Built": int(base_row["Year_Built"]),
                "Furnished_Status": base_row["Furnished_Status"],
                "Floor_No": int(base_row["Floor_No"]),
                "Total_Floors": int(base_row["Total_Floors"]),
                "Nearby_Schools": analysis_schools,
                "Nearby_Hospitals": analysis_hospitals,
                "Public_Transport_Accessibility": analysis_transport,
                "Parking_Space": base_row["Parking_Space"],
                "Security": base_row["Security"],
                "Amenities": base_row["Amenities"],
                "Facing": base_row["Facing"],
                "Owner_Type": base_row["Owner_Type"],
                "Availability_Status": base_row["Availability_Status"],
            }
            result = predict(input_data)

            r1, r2, r3 = st.columns(3)
            r1.metric("Model Market Price", format_lakhs(result["Estimated_Market_Price"]))
            r2.metric("Current Ask", format_lakhs(result["Current_Price"]))
            r3.metric("Expected Upside", f"{result['Expected_Upside_Percent']:.2f}%")

            if result["Good_Investment"] == "Yes":
                st.success("Investment signal: Good match based on model price and benchmark.")
            else:
                st.warning("Investment signal: Watchlist. Negotiate price or compare nearby options.")

        st.markdown("</div>", unsafe_allow_html=True)

    with analysis_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Location And Amenities Score</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-note">Schools, hospitals, transport access, and amenities create a practical livability score.</p>',
            unsafe_allow_html=True,
        )
        if filtered_df.empty:
            st.info("No location score to display.")
        else:
            score_df = filtered_df.head(12).copy()
            fig = px.bar(
                score_df,
                x="Infra_Score",
                y="Property_Title",
                orientation="h",
                color="Rental_Yield",
                color_continuous_scale=["#e9f7f5", "#08a8a6", "#202631"],
                labels={"Infra_Score": "Infrastructure Score", "Property_Title": ""},
            )
            fig.update_layout(
                height=435,
                margin=dict(l=5, r=5, t=5, b=5),
                yaxis={"categoryorder": "total ascending"},
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                coloraxis_colorbar_title="Yield",
            )
            st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

with tab_listings:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Property Listings</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-note">Ready-to-use shortlist table with budget, locality, nearby infrastructure, amenities, and cashflow fields.</p>',
        unsafe_allow_html=True,
    )
    if filtered_df.empty:
        st.info("No listings found for the selected filters.")
    else:
        listing_df = filtered_df[
            [
                "ID",
                "Property_Title",
                "State",
                "City",
                "Locality",
                "Property_Type",
                "BHK",
                "Size_in_SqFt",
                "Price_in_Lakhs",
                "Price_per_SqFt_Display",
                "Nearby_Schools",
                "Nearby_Hospitals",
                "Public_Transport_Accessibility",
                "Parking_Space",
                "Rental_Yield",
                "Net_Cashflow_Lakhs",
                "Availability_Status",
            ]
        ].rename(
            columns={
                "Property_Title": "Property",
                "Size_in_SqFt": "SqFt",
                "Price_in_Lakhs": "Price Lakhs",
                "Price_per_SqFt_Display": "Price/SqFt",
                "Nearby_Schools": "Schools",
                "Nearby_Hospitals": "Hospitals",
                "Public_Transport_Accessibility": "Transport",
                "Parking_Space": "Parking",
                "Rental_Yield": "Yield %",
                "Net_Cashflow_Lakhs": "Net Cashflow Lakhs/M",
                "Availability_Status": "Status",
            }
        )
        st.dataframe(listing_df, width="stretch", hide_index=True, height=560)
    st.markdown("</div>", unsafe_allow_html=True)

with tab_cashflow:
    if filtered_df.empty:
        st.info("No cashflow report available until properties match the filters.")
    else:
        cash_df = filtered_df.copy()
        cash_df["Loan_Amount_Lakhs"] = cash_df["Price_in_Lakhs"] * (1 - down_payment_pct / 100)
        cash_df["EMI_Lakhs"] = cash_df["Loan_Amount_Lakhs"].apply(lambda value: monthly_emi(value, loan_rate, loan_years)).round(2)
        cash_df["Monthly_Rent_Lakhs"] = cash_df["Estimated_Rent_Lakhs"]
        cash_df["Monthly_Net_After_EMI"] = (
            cash_df["Monthly_Rent_Lakhs"] - cash_df["Maintenance_Lakhs"] - cash_df["EMI_Lakhs"]
        ).round(2)

        cash_top, cash_chart = st.columns([1, 1.35], gap="large")
        with cash_top:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<p class="section-title">Cashflow Summary</p>', unsafe_allow_html=True)
            st.markdown(
                '<p class="section-note">Uses your loan assumptions from the sidebar.</p>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            c1.metric("Avg Monthly Rent", format_lakhs(cash_df["Monthly_Rent_Lakhs"].mean()))
            c2.metric("Avg EMI", format_lakhs(cash_df["EMI_Lakhs"].mean()))
            c3, c4 = st.columns(2)
            c3.metric("Avg Net After EMI", format_lakhs(cash_df["Monthly_Net_After_EMI"].mean()))
            c4.metric("Positive Cashflow", f"{int((cash_df['Monthly_Net_After_EMI'] > 0).sum()):,}")
            st.markdown("</div>", unsafe_allow_html=True)

        with cash_chart:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<p class="section-title">Revenue Vs Expenses</p>', unsafe_allow_html=True)
            top_cash = cash_df.head(12)
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=top_cash["Property_Title"],
                    y=top_cash["Monthly_Rent_Lakhs"],
                    name="Rent",
                    marker_color="#19b36b",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=top_cash["Property_Title"],
                    y=top_cash["EMI_Lakhs"] + top_cash["Maintenance_Lakhs"],
                    name="EMI + Maintenance",
                    marker_color="#4f75f6",
                )
            )
            fig.update_layout(
                barmode="group",
                height=365,
                margin=dict(l=5, r=5, t=5, b=5),
                legend_title_text="",
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                xaxis_tickangle=-35,
            )
            st.plotly_chart(fig, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        report_df = cash_df[
            [
                "Property_Title",
                "City",
                "Locality",
                "Price_in_Lakhs",
                "Monthly_Rent_Lakhs",
                "EMI_Lakhs",
                "Maintenance_Lakhs",
                "Monthly_Net_After_EMI",
                "Rental_Yield",
            ]
        ].rename(
            columns={
                "Property_Title": "Property",
                "Price_in_Lakhs": "Price Lakhs",
                "Monthly_Rent_Lakhs": "Rent Lakhs/M",
                "EMI_Lakhs": "EMI Lakhs/M",
                "Maintenance_Lakhs": "Maintenance Lakhs/M",
                "Monthly_Net_After_EMI": "Net After EMI Lakhs/M",
                "Rental_Yield": "Yield %",
            }
        )
        st.dataframe(report_df, width="stretch", hide_index=True, height=360)
        st.markdown("</div>", unsafe_allow_html=True)

with tab_market:
    if filtered_df.empty:
        st.info("No market analysis available for the current filters.")
    else:
        m1, m2 = st.columns(2, gap="large")
        with m1:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<p class="section-title">City Price Benchmark</p>', unsafe_allow_html=True)
            city_avg = (
                filtered_df.groupby("City", as_index=False)["Price_in_Lakhs"]
                .mean()
                .sort_values("Price_in_Lakhs", ascending=False)
            )
            fig = px.bar(
                city_avg,
                x="City",
                y="Price_in_Lakhs",
                color="Price_in_Lakhs",
                color_continuous_scale=["#e9f7f5", "#08a8a6", "#202631"],
                labels={"Price_in_Lakhs": "Avg Price Lakhs"},
            )
            fig.update_layout(height=380, margin=dict(l=5, r=5, t=5, b=5), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
            st.plotly_chart(fig, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        with m2:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<p class="section-title">Size Vs Price</p>', unsafe_allow_html=True)
            fig = px.scatter(
                filtered_df,
                x="Size_in_SqFt",
                y="Price_in_Lakhs",
                color="BHK",
                size="Infra_Score",
                hover_name="Property_Title",
                color_continuous_scale=["#08a8a6", "#4f75f6"],
                labels={"Size_in_SqFt": "Size SqFt", "Price_in_Lakhs": "Price Lakhs"},
            )
            fig.update_layout(height=380, margin=dict(l=5, r=5, t=5, b=5), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
            st.plotly_chart(fig, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">BHK And Locality Comparison</p>', unsafe_allow_html=True)
        locality_view = (
            filtered_df.groupby(["Locality", "BHK"], as_index=False)
            .agg(
                Average_Price_Lakhs=("Price_in_Lakhs", "mean"),
                Average_Yield=("Rental_Yield", "mean"),
                Properties=("ID", "count"),
            )
            .sort_values("Properties", ascending=False)
            .head(30)
        )
        st.dataframe(locality_view, width="stretch", hide_index=True, height=360)
        st.markdown("</div>", unsafe_allow_html=True)


#cd C:\Users\mikez\PycharmProjects\RealEstateAdvisor\Real_Estate_Advisor
# streamlit run app.py