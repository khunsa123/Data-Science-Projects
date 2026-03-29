# alberta_real_estate_analysis.ipynb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create small dataset
data = {
    'City': ['Calgary','Edmonton','Red Deer','Lethbridge','Medicine Hat','Calgary','Edmonton','Red Deer','Lethbridge','Medicine Hat'],
    'Price': [450000, 350000, 300000, 280000, 250000, 460000, 355000, 310000, 290000, 260000],
    'Bedrooms': [3,3,2,3,2,4,3,3,2,3],
    'Bathrooms': [2,2,1,2,1,3,2,2,1,2],
    'Year': [2023,2023,2023,2023,2023,2024,2024,2024,2024,2024]
}

df = pd.DataFrame(data)

# Average price per city
avg_price = df.groupby('City')['Price'].mean().reset_index()
plt.figure(figsize=(8,5))
sns.barplot(data=avg_price, x='City', y='Price')
plt.title("Average Housing Price per City (Alberta)")
plt.ylabel("Price (CAD)")
plt.xlabel("City")
plt.tight_layout()
plt.savefig("avg_price_city.png")
plt.show()

# Price distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Price'], bins=5, kde=True)
plt.title("Distribution of Housing Prices")
plt.xlabel("Price (CAD)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("price_distribution.png")
plt.show()

# Price trend over years
trend = df.groupby('Year')['Price'].mean().reset_index()
plt.figure(figsize=(6,4))
sns.lineplot(data=trend, x='Year', y='Price', marker='o')
plt.title("Average Price Trend Over Years")
plt.ylabel("Average Price (CAD)")
plt.xlabel("Year")
plt.tight_layout()
plt.savefig("price_trend.png")
plt.show()
