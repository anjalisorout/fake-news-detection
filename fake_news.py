# Fake News Detection using ML

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    'text': [
        'Breaking news: something happened',
        'This is fake news',
        'Government announces new policy',
        'Fake rumors spreading online'
    ],
    'label': [1, 0, 1, 0]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))

user_input = input("Enter news: ")
input_vec = vectorizer.transform([user_input])
prediction = model.predict(input_vec)

if prediction[0] == 1:
    print("Real News")
else:
    print("Fake News")
