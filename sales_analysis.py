# Customer Churn / Sales Trend Analysis - Vaibhavi Swami | IC-2026-3303
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

np.random.seed(42)
n = 1000
dates = pd.date_range('2024-01-01', '2025-12-31', freq='D')

df = pd.DataFrame({
    'OrderDate': np.random.choice(dates, n),
    'CustomerSegment': np.random.choice(['Consumer','Corporate','Home Office'], n),
    'ProductCategory': np.random.choice(['Electronics','Clothing','Furniture','Groceries'], n),
    'Sales': np.random.uniform(20, 1000, n).round(2),
    'Quantity': np.random.randint(1, 10, n),
    'Churned': np.random.choice([0,1], n, p=[0.8,0.2])
})

# 1 Feature engineering & date extraction
df['Year'] = df['OrderDate'].dt.year
df['Month'] = df['OrderDate'].dt.month
df['MonthName'] = df['OrderDate'].dt.month_name()
df['Quarter'] = df['OrderDate'].dt.quarter
df['DayOfWeek'] = df['OrderDate'].dt.day_name()
df['Revenue'] = df['Sales'] * df['Quantity']

print("Shape:", df.shape)
print(df.head())

# 2 Pivot tables
pivot_segment = pd.pivot_table(df, values='Revenue', index='CustomerSegment', aggfunc='sum').sort_values('Revenue', ascending=False)
print("\nRevenue by Customer Segment:\n", pivot_segment)

pivot_category = pd.pivot_table(df, values='Revenue', index='ProductCategory', aggfunc='sum').sort_values('Revenue', ascending=False)
print("\nRevenue by Product Category:\n", pivot_category)

churn_rate = df.groupby('CustomerSegment')['Churned'].mean().round(3)*100
print("\nChurn Rate % by Segment:\n", churn_rate)

# 3 Trend calculations
monthly_trend = df.groupby(pd.Grouper(key='OrderDate', freq='ME'))['Revenue'].sum()
print("\nMonthly Revenue Trend head:\n", monthly_trend.head())
mom_growth = monthly_trend.pct_change().mean()*100
print(f"\nAvg MoM Growth: {mom_growth:.2f}%")

# Plots
sns.set_style("whitegrid")
plt.figure(figsize=(10,5))
monthly_trend.plot(marker='o')
plt.title('Monthly Revenue Trend')
plt.ylabel('Revenue'); plt.tight_layout(); plt.show()

plt.figure(figsize=(7,5))
sns.barplot(x=pivot_segment.index, y=pivot_segment['Revenue'], palette='viridis')
plt.title('Revenue by Customer Segment')
plt.tight_layout(); plt.show()

# 4 Actionable business recommendations summary
print("\n--- BUSINESS RECOMMENDATIONS ---")
print(f"1. Top segment is {pivot_segment.index[0]} with highest revenue. Focus retention offers here.")
print(f"2. Top category is {pivot_category.index[0]}. Increase stock and promotions.")
print(f"3. Highest churn in {churn_rate.idxmax()} at {churn_rate.max()}%. Run win-back campaign.")
print(f"4. Avg monthly growth {mom_growth:.2f}%. Q4 peaks suggest festive push.")
print("5. Home Office shows lower revenue. Create B2B bundle pricing.")
