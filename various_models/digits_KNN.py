### ML with in-build digits dataset
### Basic model with KNeighborsClassifier

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the digits dataset digits
digits = datasets.load_digits()

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(f"Accuracy: {round(knn.score(X_test, y_test), 4)*100}%")

# print(digits.keys())
# print(digits.images.shape)
# print(digits.data.shape)

for x in range(0, 100):
    # index = random.randint(0, 1797)
    index = x
    pred = knn.predict(digits.data)
    dict = {}

    if pred[index] != digits.target[index]:
        print(f"Number index: {index}")
        print(f"Predicted number: {pred[index]}")
        print(f"Actual number: {digits.target[index]}")
        plt.imshow(digits.images[index], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.show()


