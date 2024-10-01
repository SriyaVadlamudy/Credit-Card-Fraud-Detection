# Importing all the dependencies
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
credit_card = pd.read_csv('dataset/creditcard 2.csv')

# To see the first 5 rows
print(credit_card.head())
# To see the last 5 rows
print(credit_card.tail())

# To check if there are any missing values in the data
credit_card.info()
credit_card.isnull().sum()

print(credit_card['Class'].value_counts())

# Dividing the data in fraud and not fraud
Not_fraud = credit_card[credit_card.Class == 0]
fraud = credit_card[credit_card.Class == 1]

print(Not_fraud)
print(fraud)

print(Not_fraud.shape)
print(fraud.shape)

counts = credit_card.Class.value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=counts.index, y=counts)
plt.title(' Fraudulent vs. Non-Fraudulent Transactions')
plt.ylabel('Count')
plt.xlabel('Class (0 or 1)')
plt.show()

# Since the dataset is unbalanced ,we will perform Under Sampling
Not_fraud_sample = Not_fraud.sample(n=492)
new_dataset = pd.concat([Not_fraud_sample, fraud], axis=0)
# When axis is 1 the dataframes will be concatenated column wise

new_dataset.head()
new_dataset.tail()
new_dataset['Class'].value_counts()
# To differentiate between the fraudulent and legit transactions
new_dataset.groupby('Class').mean()

# Splitting the dataset into X and Y
# X are the inputs,Y is the output
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# Splitting the data in training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X.shape, X_train.shape, X_test.shape)

# Using the logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)

# Training data accuracy
train_prediction = log_reg.predict(X_train)
training_data_accuracy = accuracy_score(train_prediction, Y_train)
print('Training data accuracy: ', training_data_accuracy)

# Test data accuracy
test_prediction = log_reg.predict(X_test)
test_data_accuracy = accuracy_score(test_prediction, Y_test)
print('Test Data accuracy: ', test_data_accuracy)
# OR, print(log_reg.score(X_test, Y_test))

# Confusion matrix
train_prediction = log_reg.predict(X_test)
print(confusion_matrix(Y_test, train_prediction))

# Principal Component Analysis
pca = PCA(n_components=2)
x_train = pca.fit_transform(X_train)
x_test = pca.fit_transform(X_test)

# To plot the graphs
plt.subplot(1, 2, 1)
plt.scatter(x_train[:, 0], x_train[:, 1], c=Y_train)
plt.title('Training data')
plt.xlabel('X1')
plt.ylabel("Y1")

plt.subplot(1, 2, 2)
plt.scatter(x_test[:, 0], x_test[:, 1], c=Y_test)
plt.title('Testing data')
plt.xlabel('X2')
plt.ylabel("Y2")
plt.show()


