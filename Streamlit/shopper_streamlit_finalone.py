# ========================
# Import Required Libraries
# ========================
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# ========================
# Load Dataset
# ========================
df = pd.read_csv("C:/Users/Admin/Downloads/Shopper_Spectrum_project_streamlit/Shop_Data_For_Streamlit.csv")

# ========================
# Load Models & Scaler
# ========================
saved_model = joblib.load("C:/Users/Admin/Downloads/Shopper_Spectrum_project_streamlit/kmeans_model.pkl")
scaler = joblib.load("C:/Users/Admin/Downloads/Shopper_Spectrum_project_streamlit/rfm_scaler.pkl")

# If model was saved separately (not dict), adjust:
model = saved_model  

# Drop unwanted column (CSV export index)
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# ========================
# Predict Clusters Dynamically
# ========================
def assign_clusters(df):
    if all(col in df.columns for col in ["Recency", "Frequency", "Monetary"]):
        X_scaled = scaler.transform(df[["Recency", "Frequency", "Monetary"]])
        df["Cluster"] = model.predict(X_scaled)
    else:
        st.error("❌ Dataset missing Recency, Frequency, or Monetary columns")
    return df

df = assign_clusters(df)

# ========================
# Streamlit Config
# ========================
st.set_page_config(page_title="🚖 Shopper Spectrum Dashboard", layout="wide")

# Sidebar
st.sidebar.image(
    "https://media.istockphoto.com/id/487771742/photo/concept-of-shop.jpg?s=612x612&w=0&k=20&c=7ysgREUI6wE7hAJ98jgwpp_BpVNuxiu0I8VUQL2VRkk=",
    width=100,
)
st.sidebar.title("SHOP Dashboard")

# Page Navigation
page = st.sidebar.radio("Navigate", ["Home","Product Recommendation & Segmentation", "Show Related Search Details","Show Related by Cluster","Data Analytics", "About"])

# ========================
# Home Page
# ========================
if page == "Home":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("🚖 **Welcome to the Shopper Dashboard**")
        st.markdown(
            """
            This app lets you:

            🎯 Get product recommendations  
            🧑‍🤝‍🧑 Predict customer segment  
            🔍 Explore related products  
            📊 Analyze RFM clusters with interactive graphs  

            Use the sidebar to get started.
            """
        )
    with col2:
        st.image("https://media.istockphoto.com/id/487771742/photo/concept-of-shop.jpg?s=612x612&w=0&k=20&c=7ysgREUI6wE7hAJ98jgwpp_BpVNuxiu0I8VUQL2VRkk=")

# ========================
# Product Recommendation & Customer Segmentation
# ========================
elif page == "Product Recommendation & Segmentation":
    st.header("🎯 1️⃣ Product Recommendation Module")

    product_list = sorted(df["Description"].dropna().unique().tolist())
    product_name = st.text_input("Enter a Product Name:")

    if st.button("Get Recommendations"):
        if product_name not in product_list:
            st.warning("❌ Product not found in dataset. Try another one.")
        else:
            invoices = df[df["Description"] == product_name]["InvoiceNo"].unique()

            related_products = (
                df[df["InvoiceNo"].isin(invoices) & (df["Description"] != product_name)]
                ["Description"]
                .value_counts()
                .head(5)
            )

            if related_products.empty:
                st.info("No strong co-purchase found. Try another product.")
            else:
                st.subheader("✅ Recommended Products:")
                for idx, (prod, count) in enumerate(related_products.items(), 1):
                    st.markdown(f"**{idx}. {prod}**  _(bought {count} times with {product_name})_")

    st.markdown("---")
    st.header("🧑‍🤝‍🧑 2️⃣ Customer Segmentation Module")

    recency = st.number_input("Recency (days since last purchase):", min_value=0, step=1)
    frequency = st.number_input("Frequency (number of purchases):", min_value=0, step=1)
    monetary = st.number_input("Monetary (total spend):", min_value=0.0, step=10.0)

    if st.button("Predict Cluster"):
        if recency == 0 and frequency == 0 and monetary == 0:
            st.warning("⚠️ Please enter values for Recency, Frequency, and Monetary.")
        else:
            input_scaled = scaler.transform([[recency, frequency, monetary]])
            cluster_label = model.predict(input_scaled)[0]

            cluster_names = {
                0: "💎 High-Value",
                1: "📅 Regular",
                2: "🎯 Occasional",
                3: "⚠️ At-Risk",
            }
            st.success(f"Predicted Cluster: **Cluster {cluster_label} - {cluster_names.get(cluster_label, 'Unknown')}**")

# ========================
# Show Related Search Details Page
# ========================
elif page == "Show Related Search Details":
    st.header("🔎 Search Product & Find Related Items")

    def search_product(product_name, df):
        matches = df[df["Description"].str.contains(product_name, case=False, na=False)]
        if matches.empty:
            st.warning(f"❌ No product found matching: {product_name}")
            return None

        cluster_counts = matches["Cluster"].value_counts()
        st.success(f"✅ Product search: '{product_name}'")
        st.write("Clusters where this product appears:")
        st.write(cluster_counts)

        top_cluster = cluster_counts.index[0]
        related_products = (
            df[df["Cluster"] == top_cluster]["Description"]
            .value_counts()
            .head(10)
        )
        st.write(f"🔹 Related Products in Cluster {top_cluster}:")
        st.write(related_products)

        return matches[["Description", "Quantity", "UnitPrice", "Total_Amount", "Cluster"]]

    search = st.text_input("Enter Product Name:")
    if search:
        results = search_product(search, df)
        if results is not None:
            st.write("🔹 Sample Matching Rows:")
            st.dataframe(results.head())

# ========================
# Show Related by Cluster Page
# ========================
elif page == "Show Related by Cluster":
    st.header("🔎 Explore Related Products by Cluster")

    cluster_input = st.number_input(
        "Enter Cluster Number:",
        min_value=int(df["Cluster"].min()),
        max_value=int(df["Cluster"].max()),
        step=1,
    )

    if st.button("Show Related Products"):
        cluster_data = df[df["Cluster"] == cluster_input]
        if cluster_data.empty:
            st.warning(f"❌ No products found in Cluster {cluster_input}")
        else:
            st.success(f"✅ Showing products from Cluster {cluster_input}")
            top_products = (
                cluster_data["Description"]
                .value_counts()
                .head(10)
                .reset_index()
                .rename(columns={"index": "Product", "Description": "Count"})
            )
            st.subheader("📦 Top 10 Products in this Cluster")
            st.dataframe(top_products)

# ========================
# Data Analytics Page
# ========================
elif page == "Data Analytics":
    st.title("📊 Shopper Spectrum Data - Relationship Analysis")

    col_controls, col_plots = st.columns([1, 3])

    with col_controls:
        st.header("⚙️ Plot Options")
        relation_plot = st.selectbox(
            "Choose a relation to visualize:",
            [
                "Cluster Distribution",
                "Average RFM by Cluster",
                "Monetary vs Frequency (colored by Cluster)",
                "Recency vs Frequency (colored by Cluster)",
                "Quantity vs Total Amount (colored by Cluster)",
                "Top Products by Revenue",
                "Top Products by Quantity (High Qty)",
                "Bottom Products by Quantity (Low Qty)",
                "Top Products by Average Unit Price",
                "Average Unit Price by Cluster",
            ],
        )
        st.markdown("---")

    with col_plots:
        fig = None
        if relation_plot == "Cluster Distribution":
            cluster_counts = df["Cluster"].value_counts().sort_index()
            fig = px.pie(values=cluster_counts.values, names=cluster_counts.index,
                         title="Customer Segments Distribution", hole=0.4)

        elif relation_plot == "Average RFM by Cluster":
            rfm_means = df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().reset_index()
            fig = px.bar(rfm_means, x="Cluster", y=["Recency", "Frequency", "Monetary"],
                         barmode="group", title="Average RFM by Cluster")

        elif relation_plot == "Monetary vs Frequency (colored by Cluster)":
            fig = px.scatter(df, x="Frequency", y="Monetary", color="Cluster",
                             title="Monetary vs Frequency")

        elif relation_plot == "Recency vs Frequency (colored by Cluster)":
            fig = px.scatter(df, x="Recency", y="Frequency", color="Cluster",
                             title="Recency vs Frequency")

        elif relation_plot == "Quantity vs Total Amount (colored by Cluster)":
            fig = px.scatter(df, x="Quantity", y="Total_Amount", color="Cluster",
                             title="Quantity vs Total Amount")

        elif relation_plot == "Top Products by Revenue":
            top_products = df.groupby("Description")["Total_Amount"].sum().nlargest(10).reset_index()
            fig = px.bar(top_products, x="Description", y="Total_Amount",
                         title="Top 10 Products by Revenue")

        elif relation_plot == "Top Products by Quantity (High Qty)":
            top_products_qty = df.groupby("Description")["Quantity"].sum().nlargest(10).reset_index()
            fig = px.bar(top_products_qty, x="Description", y="Quantity",
                         title="Top 10 Products by Quantity Sold")

        elif relation_plot == "Bottom Products by Quantity (Low Qty)":
            low_products_qty = df.groupby("Description")["Quantity"].sum().nsmallest(10).reset_index()
            fig = px.bar(low_products_qty, x="Description", y="Quantity",
                         title="Bottom 10 Products by Quantity Sold")

        elif relation_plot == "Top Products by Average Unit Price":
            top_products_price = df.groupby("Description")["UnitPrice"].mean().nlargest(10).reset_index()
            fig = px.bar(top_products_price, x="Description", y="UnitPrice",
                         title="Top 10 Products by Average Unit Price")

        elif relation_plot == "Average Unit Price by Cluster":
            avg_price = df.groupby("Cluster")["UnitPrice"].mean().reset_index()
            fig = px.bar(avg_price, x="Cluster", y="UnitPrice", title="Average Unit Price by Cluster")

        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.sidebar.success("✅ Select any relationship graph to view insights.")

# ========================
# About Page
# ========================
elif page == "About":
    st.header("📚 About This App")
    st.write("This dashboard is built using **Streamlit**.")
    st.write("**Data Source**: Online Retail Dataset (preprocessed with RFM and KMeans clustering)")
    st.markdown("---")
    st.write("Developed by **Arun Kumar**")
    st.caption("Thank you for visiting!")

# ========================
# Footer
# ========================
st.markdown("---")
st.caption("Developed by Arun Kumar | Powered by GUVI")
