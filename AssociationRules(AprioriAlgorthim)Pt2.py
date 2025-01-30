# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:39:04 2024

@author: Armanis
"""
#%%
# Import necessary libraries
from pathlib import Path
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\Armanis\OneDrive\Desktop\Python CSV Files\Coursetopics.csv"  # Adjust path as necessary
catalog = pd.read_csv(file_path)

# Look at the data structure
print("Dataset preview:")
print(catalog.head())
print("Columns in dataset:")
print(catalog.columns)


#%%

# Analyze item frequencies
item_frequency = catalog.sum(axis=0) / len(catalog) #  computes the total number of customers who selected each course(column wise) / total # of costumers (all rows combined)
print("\nItem Frequencies (Relative):")
print(item_frequency)

# Plot item frequencies
ax = item_frequency.plot.bar(color='blue', figsize=(10, 6))
plt.title("Relative Item Frequencies")
plt.ylabel("Relative Frequency")
plt.xlabel("Courses")
plt.tight_layout()
plt.show()


#%%
# Generate frequent itemsets
itemsets = apriori(catalog, min_support=0.05, use_colnames=True) #specifies the minimum proportion of transactions in which an itemset must appear to be considered frequent. It helps filter out infrequent itemsets and focus on meaningful patterns.
print("\nFrequent Itemsets:")
print(itemsets)

# Manually calculate num_itemsets (if needed)
num_itemsets = len(itemsets)

# Generate association rules
rules = association_rules(itemsets, metric="confidence", min_threshold=0.1, num_itemsets=num_itemsets)
print(f"\nNumber of rules generated: {len(rules)}")

# Sort and display top 10 rules
sorted_rules = rules.sort_values(by="lift", ascending=False).head(10)
print("\nTop 10 Association Rules Sorted by Lift:")
print(sorted_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Interpret rules
print("\nRule Interpretations:")
for idx, rule in sorted_rules.iterrows():
    antecedent = ', '.join(list(rule['antecedents']))
    consequent = ', '.join(list(rule['consequents']))
    print(f"If a customer purchases {antecedent}, they are likely to purchase {consequent}.")
    print(f" - Support: {rule['support']:.2f}, Confidence: {rule['confidence']:.2f}, Lift: {rule['lift']:.2f}\n")


