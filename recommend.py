# recommend.py

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def generate_rules():

    df = pd.read_csv("cleaned_retail.csv")

    # Keep only top 20 products (to avoid memory crash)
    top_products = df['Description'].value_counts().head(20).index
    df = df[df['Description'].isin(top_products)]

    transactions = df.groupby("InvoiceNo")["Description"].apply(list).tolist()

    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(te_array, columns=te.columns_)

    frequent_itemsets = apriori(
        basket,
        min_support=0.01,
        use_colnames=True,
        max_len=2
    )

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.3
    )

    rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

    rules.to_csv("rules.csv", index=False)

    return rules


import os

def recommend(selected_product):

    import os

    if not os.path.exists("rules.csv"):
        generate_rules()

    rules = pd.read_csv("rules.csv")

    rules['antecedents'] = rules['antecedents'].apply(eval)
    rules['consequents'] = rules['consequents'].apply(eval)

    # Single item antecedents only
    rules = rules[rules['antecedents'].apply(lambda x: len(x) == 1)]

    results = rules[
        rules['antecedents'].apply(lambda x: selected_product in x)
    ]

    results = results[
        (results['lift'] > 1) &
        (results['confidence'] > 0.3)
    ]

    results = results.sort_values(by="confidence", ascending=False)

    recommended_products = []

    for itemset in results['consequents']:
        for item in itemset:
            if item != selected_product:
                recommended_products.append(item)

    recommended_products = list(dict.fromkeys(recommended_products))

    # 🎯 Fallback: if no strong rules found
    if len(recommended_products) == 0:

        df = pd.read_csv("cleaned_retail.csv")
        popular_products = (
            df['Description']
            .value_counts()
            .index.tolist()
        )

        recommended_products = [
            item for item in popular_products
            if item != selected_product
        ][:3]

    return pd.DataFrame({
        "Recommended Product": recommended_products[:3]
    })
