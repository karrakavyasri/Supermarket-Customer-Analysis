import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("supermarket_customers.csv")

# Create figure with 3 subplots
plt.figure(figsize=(18, 5))

# -------------------------------
# GRAPH 1: CUSTOMER SEGMENTATION
# -------------------------------
X = data[['Annual_Income', 'Spending_Score']]
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

plt.subplot(1, 3, 1)
plt.scatter(
    data['Annual_Income'],
    data['Spending_Score'],
    c=data['Cluster']
)
plt.title("Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

# -------------------------------
# GRAPH 2: BUYING PATTERNS
# -------------------------------
plt.subplot(1, 3, 2)
product_counts = data['Product'].value_counts()
product_counts.plot(kind='bar')
plt.title("Product Buying Pattern")
plt.xlabel("Product")
plt.ylabel("Purchases")

# -------------------------------
# GRAPH 3: PRODUCT COMBINATIONS
# -------------------------------
plt.subplot(1, 3, 3)
basket = pd.crosstab(data['CustomerID'], data['Product'])

# ✅ FIXED LINE (NO ERROR)
basket = (basket > 0).astype(int)

plt.imshow(basket.corr())
plt.colorbar()
plt.xticks(range(len(basket.columns)), basket.columns)
plt.yticks(range(len(basket.columns)), basket.columns)
plt.title("Product Combination")

plt.tight_layout()
plt.show()
