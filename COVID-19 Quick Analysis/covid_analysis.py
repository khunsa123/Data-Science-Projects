# covid19_analysis.ipynb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset directly from GitHub
url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
df = pd.read_csv(url)

# Keep only useful columns
df = df[['location', 'date', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths']]

# Filter for Canada and last 60 days
canada_df = df[df['location'] == 'Canada'].tail(60)

# Plot daily new cases for Canada
plt.figure(figsize=(10,5))
sns.lineplot(data=canada_df, x='date', y='new_cases', marker='o')
plt.xticks(rotation=45)
plt.title("Canada: Daily New COVID-19 Cases (Last 60 Days)")
plt.ylabel("New Cases")
plt.xlabel("Date")
plt.tight_layout()
plt.savefig("canada_new_cases.png")
plt.show()

# Top 5 countries by total cases
latest_df = df[df['date'] == df['date'].max()]
top5 = latest_df.nlargest(5, 'total_cases')[['location', 'total_cases']]

plt.figure(figsize=(8,5))
sns.barplot(data=top5, x='location', y='total_cases')
plt.title("Top 5 Countries by Total COVID-19 Cases")
plt.ylabel("Total Cases")
plt.xlabel("Country")
plt.tight_layout()
plt.savefig("top5_countries_cases.png")
plt.show()
