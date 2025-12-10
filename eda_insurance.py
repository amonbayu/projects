import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Visual style
plt.style.use("ggplot")
sns.set_theme()

# 1. Load Dataset

df = pd.read_csv("insurance_cleaned.csv")







#Distribution plots
def plot_hist(column, color="blue"):
    plt.figure(figsize=(6,4))
    sns.histplot(df[column], kde=True, color=color)
    plt.title(f"{column} Distribution")
    plt.tight_layout()
    plt.show()

plot_hist("age")
plot_hist("bmi", color="purple")
plot_hist("charges", color="darkgreen")

#boxplots for numerical features
plt.figure(figsize=(14,4))

plt.subplot(1,3,1)
sns.boxplot(y=df["age"])
plt.title("Age")

plt.subplot(1,3,2)
sns.boxplot(y=df["bmi"])
plt.title("BMI")

plt.subplot(1,3,3)
sns.boxplot(y=df["charges"])
plt.title("Charges")

plt.tight_layout()
plt.show()


# correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# Pairplot for Numerical Features
sns.pairplot(df[['age', 'bmi', 'children', 'charges']])
plt.show()



# Smoker vs Charges
plt.figure(figsize=(6,4))
sns.barplot(x="smoker_yes", y="charges", data=df)
plt.title("Average Charges: Smoker vs Non-Smoker")
plt.xticks([0,1], ["Non-Smoker", "Smoker"])
plt.tight_layout()
plt.show()

# Sex vs Charges
plt.figure(figsize=(6,4))
sns.barplot(x="sex_male", y="charges", data=df)
plt.title("Average Charges: Male vs Female")
plt.xticks([0,1], ["Female", "Male"])
plt.tight_layout()
plt.show()


#Region vs Charges
region_cols = ["region_northwest", "region_southeast", "region_southwest"]

region_avg = {
    "northeast": df[df[region_cols].sum(axis=1) == 0]["charges"].mean(),
    "northwest": df[df["region_northwest"] == 1]["charges"].mean(),
    "southeast": df[df["region_southeast"] == 1]["charges"].mean(),
    "southwest": df[df["region_southwest"] == 1]["charges"].mean(),
}

plt.figure(figsize=(7,4))
pd.Series(region_avg).plot(kind="bar", color="teal")
plt.title("Average Charges by Region")
plt.ylabel("Mean Charges")
plt.tight_layout()
plt.show()

# BMI vs Charges
plt.figure(figsize=(6,4))
sns.scatterplot(x="bmi", y="charges", hue="smoker_yes", data=df)
plt.title("BMI vs Charges (Colored by Smoker Status)")
plt.tight_layout()
plt.show()

# Age vs Charges
plt.figure(figsize=(6,4))
sns.scatterplot(x="age", y="charges", hue="smoker_yes", data=df)
plt.title("Age vs Charges")
plt.tight_layout()
plt.show()


print("EDA Complete.")
