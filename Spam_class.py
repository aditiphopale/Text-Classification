# Spam Classification by Aditi Phopale and Sejal wadekar

# # Importing the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


spam_classifier = pd.read_csv("spam_dataset.csv")
spam_classifier.head() #First 4 observatioons of the data
len(spam_classifier) #Length of the data


# # WordCloud
from wordcloud import WordCloud

# Wordcloud for Spam
spam_list = spam_classifier[spam_classifier["Category"] == "spam"]["Message"].unique().tolist()
spam = " ".join(spam_list)
spam_wordcloud = WordCloud().generate(spam)
plt.figure(figsize=(8,8))
plt.title("Spam Words", fontsize=20)
plt.imshow(spam_wordcloud)
plt.show()

# Wordcloud for Ham
ham_list = spam_classifier[spam_classifier["Category"] == "ham"]["Message"].unique().tolist()
ham = " ".join(ham_list)
ham_wordcloud = WordCloud().generate(ham)
plt.figure(figsize=(8,8))
plt.title("Ham Words", fontsize=20)
plt.imshow(ham_wordcloud)
plt.show()


# # Text Classification Pipeline

texts = []
labels = []
for i, label in enumerate(spam_classifier['Category']):
    texts.append(spam_classifier['Message'][i])
    if label == 'ham':
        labels.append(0)
    else:
        labels.append(1)

#Converting Input to an array
texts = np.asarray(texts) 
labels = np.asarray(labels)

#Prints the number of texts and labels
print("number of texts :" , len(texts)) 
print("number of labels: ", len(labels)) #Either spam or ham
#print(type(texts)) #Multidimensional array of fixed size items


# # We notice that this is an Imbalanced data

#Checking for the number of spam and ham observations
#print(np.unique(labels))
#print(np.bincount(labels))


# Number of 'ham' observations: 4825
# Number of 'spam' observations: 747

# # Splitting into Train and Test datasets
np.random.seed(42)
# shuffle the data
indices = np.arange(spam_classifier.shape[0])
np.random.shuffle(indices)
texts = texts[indices]
labels = labels[indices]

# we will use 80% of data as training, 20% as validation data
training_samples = int(5572 * .8)
validation_samples = int(5572 - training_samples)
# sanity check
#print(len(texts) == (training_samples + validation_samples))
print("Training samples: {0}, Validation Samples: {1} ".format(training_samples, validation_samples))

texts_train = texts[:training_samples]
y_train = labels[:training_samples]
texts_test = texts[training_samples:]
y_test = labels[training_samples:]


# # CountVectorizer

#Importing CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
vector = CountVectorizer().fit(texts_train) #Counts the number of times each word appears in document- texts_train

#Transform document to a document term matrix
X_train = vector.transform(texts_train)
#print(repr(X_train))

#Transform document to a document term matrix
X_test = vector.transform(texts_test)
#print(repr(X_test))


le = LabelEncoder()
label = le.fit_transform(spam_classifier.Category)


# # Models used: 
# 		 1. Logistic Regression
# 		 2. Decison Trees
# 		 3. Random Forest Classifier



# # Logistic Regression

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver = 'liblinear')
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(logreg, param_grid, cv=5)
logreg_train = grid.fit(X_train, y_train)


#Calculating the accuracy of test dataset
pred_logreg = logreg_train.predict(X_test)
print("\n\nAccuracy using Logistic Regression is: ", grid.score(X_test, y_test))


# # Decision Tree

#Setting the vocab
vector.fit(spam_classifier)
# convert text to vectors
X = vector.transform(spam_classifier.Message)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(spam_classifier.Category)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
dec_clf = DecisionTreeClassifier(criterion='gini')

# Fitting the model
dec_clf.fit(X_train, y_train)
preds = dec_clf.predict(X_test)
preds.shape

#Calculating Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, preds)
print("Accuracy using Decision Tree is: ", accuracy)


# # Random Forrest Classifier

from sklearn.ensemble import RandomForestClassifier
rand_clf = RandomForestClassifier(n_estimators=10)

#Fitting the model
rand_clf.fit(X_train, y_train)

#Predicting on the test dataset
preds2 = rand_clf.predict(X_test)
preds2.shape

#Calculating Accuracy
from sklearn.metrics import accuracy_score
accuracy2 = accuracy_score(y_test, preds2)
print("Accuracy using Random Forest Classifier is: ", accuracy2)

