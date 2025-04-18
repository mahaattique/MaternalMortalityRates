import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

file_path = "/workspaces/MaternalMortalityRates/BirthData.csv"
figures_folder = "/workspaces/MaternalMortalityRates/figures"
os.makedirs(figures_folder, exist_ok=True)

df = pd.read_csv(file_path, sep='\t', na_filter=True)
df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", '')

print(df.columns)
print("\nTotal rows in dataset:", len(df))
print("Total unique rows in dataset:", len(df.drop_duplicates()))


unique_races = df["Mothers Single Race 6"].unique()
print("\nUnique races in 'Mother's Single Race 6':")
print(unique_races)

race_counts = df["Mothers Single Race 6"].value_counts(dropna=False)
print("\nFrequency of each race:")
print(race_counts)

# Data info
print("\nData info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())


numeric_columns = [
    "% of Total Births", 
    "Average Age of Mother (years)", 
    "Average OE Gestational Age (weeks)", 
    "Average LMP Gestational Age (weeks)", 
    "Average Birth Weight (grams)", 
    "Average Pre-pregnancy BMI", 
    "Average Number of Prenatal Visits", 
    "Average Interval Since Last Live Birth (months)", 
    "Average Interval Since Last Other Pregnancy Outcome (months)"
]

missing_columns = [col for col in numeric_columns if col not in df.columns]
if missing_columns:
    print(f"\nWarning: These columns are missing or misnamed in the dataset: {missing_columns}")
    numeric_columns = [col for col in numeric_columns if col not in missing_columns]


df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=numeric_columns, how='all')

df['Tobacco Use'] = df['Tobacco Use'].str.strip().replace({
    'Yes': 'User',
    'No': 'Non-User',
    'Unknown or Not Stated': 'Unknown',
    'Not Available': 'Unknown',
    'Not Reported': 'Unknown'
})

# 1. Bar Chart: Mother's Race vs Baby's Birth Weight
plt.figure(figsize=(14, 7))
race_order = ['White', 'Black or African American', 'Asian', 
              'American Indian or Alaska Native',
              'Native Hawaiian or Other Pacific Islander',
              'More than one race']

ax = sns.barplot(
    x="Mothers Single Race 6",
    y="Average Birth Weight (grams)",
    data=df,
    order=race_order,
    errorbar=None,
    palette="viridis",
    estimator='mean'
)

plt.title("Average Birth Weight by Mother's Race", fontsize=16, pad=20)
plt.xlabel("Mother's Race", fontsize=14)
plt.ylabel("Average Birth Weight (grams)", fontsize=14)
plt.xticks(rotation=45, ha='right')

for p in ax.patches:
    ax.annotate(f"{p.get_height():.0f}", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 10), 
                textcoords='offset points')

plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'BirthWeight_by_Race_BarChart.png'), dpi=300)
plt.close()

race_palette = {
    'White': '#1f77b4',
    'Black or African American': '#ff7f0e',
    'Asian': '#2ca02c',
    'American Indian or Alaska Native': '#d62728',
    'Native Hawaiian or Other Pacific Islander': '#9467bd',
    'More than one race': '#8c564b'
}

known_races = [race for race in race_palette.keys() if race in df['Mothers Single Race 6'].unique()]
filtered_df = df[df['Mothers Single Race 6'].isin(known_races)]

# 2. Mother's Age Distribution by Race 
plt.figure(figsize=(14, 8))
for race in known_races:
    race_data = filtered_df[filtered_df['Mothers Single Race 6'] == race]['Average Age of Mother (years)']
    sns.kdeplot(race_data, 
                color=race_palette[race], 
                label=race,
                linewidth=2,
                alpha=0.7)

plt.xlim(10, 50) 
plt.ylim(0, 0.05)  
plt.title("Distribution of Mother's Age at Birth by Race", fontsize=16)
plt.xlabel("Age (years)", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(title="Race")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'Age_Distribution_Bell_Curves.png'), dpi=300)
plt.close()


# 3. Tobacco Use Analysis
birth_weight_stats = df.groupby('Tobacco Use')['Average Birth Weight (grams)'].agg(
    ['mean', 'count']
).rename(columns={
    'mean': 'Mean',
    'count': 'Count'
}).reindex(['Non-User', 'User', 'Unknown'])

pd.set_option('display.float_format', '{:.2f}'.format)
print("\nBirth Weight Statistics by Tobacco Use:")
print(birth_weight_stats)

# 4. Statistical Analysis
tukey_data = df[['Average Birth Weight (grams)', 'Tobacco Use']].dropna()
groups = {status: tukey_data[tukey_data['Tobacco Use'] == status]['Average Birth Weight (grams)'] 
          for status in ['Non-User', 'User', 'Unknown']}

f_val, p_val = stats.f_oneway(*groups.values())

print("\nANOVA Results for Birth Weight by Tobacco Use:")
print(f"F-value: {f_val:.2f}")
print(f"P-value: {p_val:.4f}")

# Post-hoc tests if significant and we have data
if p_val < 0.05 and len(tukey_data) > 0:
    print("\nPost-hoc Tukey HSD Results:")
    tukey = pairwise_tukeyhsd(
        endog=tukey_data['Average Birth Weight (grams)'],
        groups=tukey_data['Tobacco Use'],
        alpha=0.05
    )
    print(tukey.summary())
elif p_val >= 0.05:
    print("\nANOVA not significant - no post-hoc tests needed")
else:
    print("\nNot enough valid data for post-hoc tests")

print("\nAnalysis complete. All visualizations saved to:", figures_folder)


#5. Correlation Heatmap (with validated variables)
heatmap_vars = [
    'Average Birth Weight (grams)',
    'Average Age of Mother (years)',
    'Average Pre-pregnancy BMI',
    'Average OE Gestational Age (weeks)',
    'Average Number of Prenatal Visits'
]

# Filter to only include columns with valid data
valid_vars = [col for col in heatmap_vars if col in df.columns and df[col].notna().any()]

plt.figure(figsize=(10, 8))
sns.heatmap(
    df[valid_vars].corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    vmin=-1,
    vmax=1,
    center=0,
    linewidths=0.5,
    annot_kws={"size": 10},
    square=True
)
plt.title("Correlation Matrix of Maternal Health Factors (Clean Data)", fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'MH_Correlation.png'), dpi=300)
plt.close()




# Analysis for birth weight categories
def categorize_birth_weight(birth_weight):
    if pd.isna(birth_weight):
        return 'Unknown'
    elif birth_weight > 4500:
        return 'High Birth Weight'
    elif 1 < birth_weight < 2500:
        return 'Low Birth Weight'
    else:
        return 'Normal Birth Weight'

df['Birth Weight Category'] = df['Average Birth Weight (grams)'].apply(categorize_birth_weight)

# Group by race and birth weight category
category_counts = df.groupby(['Mothers Single Race 6', 'Birth Weight Category']).size().reset_index(name='Count')

# Bar plot: Count of each birth weight category by race
plt.figure(figsize=(14, 7))
sns.barplot(
    x='Mothers Single Race 6',
    y='Count',
    hue='Birth Weight Category',
    data=category_counts,
    palette='Set2'
)

plt.title("Birth Weight Categories by Mother's Race", fontsize=16)
plt.xlabel("Mother's Race", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'BirthWeight_Categories_by_Race.png'), dpi=300)
plt.close()


# Create a new column for birth weight categories
def categorize_birth_weight(birth_weight):
    if pd.isna(birth_weight):
        return 'Unknown'
    elif birth_weight > 4500:
        return 'High Birth Weight'
    elif  1 < birth_weight < 2500:
        return 'Low Birth Weight'
    else:
        return 'Normal Birth Weight'

df['Birth Weight Category'] = df['Average Birth Weight (grams)'].apply(categorize_birth_weight)

# Filter out rows with "Not Available" and "Not Reported"
df_filtered = df[~df['Mothers Single Race 6'].isin(['Not Available', 'Not Reported'])]


# Set up the categories
categories = ['Low Birth Weight', 'Normal Birth Weight', 'High Birth Weight']

# Create separate bar plots for each birth weight category
for category in categories:
    category_data = df_filtered[df_filtered['Birth Weight Category'] == category]
    
    # Group by race and count the occurrences of each category
    category_counts = category_data.groupby('Mothers Single Race 6').size().reset_index(name='Count')
    
    # Create a bar plot for the current category
    plt.figure(figsize=(14, 7))
    sns.barplot(
        x='Mothers Single Race 6',
        y='Count',
        data=category_counts,
        palette='Set2'
    )
    
    # Title and labels
    plt.title(f"{category} by Mother's Race", fontsize=16)
    plt.xlabel("Mother's Race", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, f'{category.replace(" ", "_")}_by_Race.png'), dpi=300)
    plt.close()


# Group by birth weight category and race, then count the occurrences
category_counts = df_filtered.groupby(['Birth Weight Category', 'Mothers Single Race 6']).size().reset_index(name='Count')

# Create the bar plot
plt.figure(figsize=(14, 7))
sns.barplot(
    x='Birth Weight Category',
    y='Count',
    hue='Mothers Single Race 6',
    data=category_counts,
    palette='Set2'
)

# Title and labels
plt.title("Birth Weight Categories by Race", fontsize=16)
plt.xlabel("Birth Weight Category", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(rotation=0)  # No rotation needed for x-axis labels

# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'BirthWeight_Categories_by_Race_ColorCoded.png'), dpi=300)
plt.close()