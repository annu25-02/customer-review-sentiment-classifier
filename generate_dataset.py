"""
Generates a labelled demo dataset (data/reviews.csv) so the project runs offline.

NOTE: This is SYNTHETIC data built from templates. It is good for learning the
pipeline, but accuracy on it will be unrealistically high. For a real project,
replace data/reviews.csv with the IMDb dataset or a real product-review dataset
(see README, "Using a real dataset").
"""
import random
import pandas as pd

random.seed(42)

products = ["product", "phone", "laptop", "jacket", "blender", "service", "delivery",
            "headphones", "app", "shoes", "mattress", "camera", "course", "restaurant"]

pos_adj = ["excellent", "amazing", "fantastic", "wonderful", "great", "superb",
           "outstanding", "perfect", "brilliant", "impressive"]
neg_adj = ["terrible", "awful", "horrible", "disappointing", "poor", "useless",
           "pathetic", "dreadful", "worthless", "unacceptable"]

pos_templates = [
    "The {p} is {a}", "I love this {p}", "This {p} is {a} and works great",
    "{a} {p}, totally worth the money", "Very happy with the {p}",
    "The quality of the {p} is {a}", "Highly recommend this {p}",
    "Best {p} I have ever bought", "I am extremely satisfied with the {p}",
    "The {p} exceeded my expectations", "Great value, the {p} is {a}",
    "Fast delivery and the {p} is {a}", "Absolutely loved the {p}, will buy again",
    "The {p} is not bad at all, actually {a}", "Never been happier with a {p}",
    "Amazing experience, the {p} was {a}", "Works perfectly, the {p} is {a}",
]
neg_templates = [
    "The {p} is {a}", "I hate this {p}", "This {p} is {a} and stopped working",
    "{a} {p}, complete waste of money", "Very unhappy with the {p}",
    "The quality of the {p} is {a}", "Do not buy this {p}",
    "Worst {p} I have ever bought", "I am extremely disappointed with the {p}",
    "The {p} did not meet my expectations", "Poor value, the {p} is {a}",
    "Late delivery and the {p} is {a}", "Regret buying the {p}, never again",
    "The {p} is not good at all", "The {p} is not worth the price",
    "Terrible experience, the {p} was {a}", "Broke after two days, the {p} is {a}",
    "The {p} is not great and I want a refund",
]

rows = []
for _ in range(700):
    p = random.choice(products)
    if random.random() < 0.5:
        rows.append((random.choice(pos_templates).format(p=p, a=random.choice(pos_adj)), "Positive"))
    else:
        rows.append((random.choice(neg_templates).format(p=p, a=random.choice(neg_adj)), "Negative"))

df = pd.DataFrame(rows, columns=["review", "sentiment"]).drop_duplicates().reset_index(drop=True)
df.to_csv("data/reviews.csv", index=False)
print(f"Saved data/reviews.csv with {len(df)} unique reviews")
print(df["sentiment"].value_counts())
