import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Evaluate the model using confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report


# Load the dataset
url = "churndata.csv"
df = pd.read_csv(url)
df.head()


# Check for missing values
df.isnull().sum()
df.dropna(inplace=True)


# Check and remove duplicates
df.duplicated().sum()

df.drop_duplicates(inplace=True)



# Convert categorical features into dummy/indicator variables
df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

# Define X and y
X = df.drop(["Exited", "Row_number", "Customer_ID", "Surname"], axis=1)
y = df["Exited"]



# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Setup ANN
model = Sequential()
model.add(Dense(units=6, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=6, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the ANN model
model.fit(X_train, y_train, batch_size=32, epochs=50)


# Predict the test set results
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Check training accuracy
train_acc = model.evaluate(X_train, y_train, verbose=0)
print(f"Training Accuracy: {train_acc[1] * 100:.2f}%")

# Check test accuracy
test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc[1] * 100:.2f}%")
