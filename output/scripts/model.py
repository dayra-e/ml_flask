# modelo de machine learning
import pandas as pd
import joblib

from sklearn.naive_bayes import GaussianNB

# cargar datos
df = pd.read_csv('data/data.csv', sep=';')
#print(df.head())

X = df[["Height", "Weight"]]
y = df["Species"]

# entrenar modelo
clf = GaussianNB()
clf.fit(X, y)

# guardar modelo
joblib.dump(clf, 'output/model.pkl')