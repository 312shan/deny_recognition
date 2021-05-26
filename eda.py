import pandas as pd

# df = pd.read_csv("data/train.tsv", sep="\t")
# print(df.head())
# print(df.columns)
# print(df.head())
# print(df.iloc[1, :]["Question"])


from sklearn.metrics import classification_report

y_true = [0, 1, 0, 1, 1]
y_pred = [0, 0, 1, 0, 1]
report = classification_report(y_true, y_pred, output_dict=True)["weighted avg"]
p, r, f1 = report["precision"], report["recall"], report["f1-score"]
print(p, r, f1)
