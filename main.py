import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
df = pd.read_csv(url)

# Drop the missing values
df = df.dropna()

# Isolating X and Y

# X : Features
# Y : Target

X = df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
y = df["species"]

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model using DecisionTreeClassifer
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Testing the model
y_pred = clf.predict(X_test)

score = accuracy_score(y_test, y_pred)
print(score)

# A hypothetical penguin to further test the model
mystery_penguin = pd.DataFrame([[45.0, 15.0, 210.0, 4500.0]], columns=X.columns)

mystery_pred = clf.predict(mystery_penguin)
print(mystery_pred) # Answer should be Gentoo