import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle


def target_encode(val):
    return target_mapper[val]


target_mapper = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}

penguins = pd.read_csv("inputs\\penguins_cleaned.csv")
df = penguins.copy()
target = "species"
encode = ["sex", "island"]

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

df["species"] = df["species"].apply(target_encode)

X = df.drop("species", axis=1)
y = df["species"]

clf = RandomForestClassifier()
clf.fit(X, y)

pickle.dump(clf, open("models\\penguins_clf.pkl", "wb"))
