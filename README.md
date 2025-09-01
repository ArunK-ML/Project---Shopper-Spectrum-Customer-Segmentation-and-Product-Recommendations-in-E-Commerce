# Project---Shopper-Spectrum-Customer-Segmentation-and-Product-Recommendations-in-E-Commerce
The global e-commerce industry generates vast amounts of transaction data daily, offering valuable insights into customer purchasing behaviors. Analyzing this data is essential for identifying meaningful customer segments and recommending relevant products to enhance customer experience and drive business growth. 

📊 **Dataset Overview**

Rows: 541,909 (transactions/records)

Columns: 8

Size: ~33 MB

📂 **Columns Description**

InvoiceNo (object) – Invoice number, unique for each transaction. If it starts with "C", it usually indicates a cancellation.

StockCode (object) – Product/item code.

Description (object) – Name/description of the product. (Some missing values: ~1,454).

Quantity (int64) – Number of units purchased. Negative values may indicate returns.

InvoiceDate (object) – Date and time of transaction (string format, needs conversion to datetime for analysis).

UnitPrice (float64) – Price of a single unit (in GBP).

CustomerID (float64) – Unique ID of customer (missing for ~135k rows).

Country (object) – Country of the customer. Total 38 countries, with United Kingdom dominating.

📈 **Quick Insights**

Top Product: WHITE HANGING HEART T-LIGHT HOLDER (most frequently purchased).

Top Country: United Kingdom (495,478 transactions).

Avg Quantity per purchase: ~9.55 items.

Avg Unit Price: £4.61.

**Missing Data:**

Description (~0.3%)

CustomerID (~25%)

This dataset is typically used for market basket analysis, RFM (Recency, Frequency, Monetary) analysis, customer segmentation, and sales forecasting.
