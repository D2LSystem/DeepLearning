# -*- coding: utf-8 -*-
# Demo file for Spyder Tutorial
# Hans Fangohr, University of Southampton, UK


import numpy as np
import pandas as pd

"""
df = pd.DataFrame(np.arange(12).reshape(3,4))
a = df.shape
print df
print a[0]
print a[1]
def hello():
    Print "Hello World" and return None
    print("Hello World")
hello()

index = pd.date_range('1/1/2000', periods=8)

df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=['A', 'B', 'C'])

df.columns = [x.lower() for x in df.columns]

#print df

print (df[:2] > 1).all"""

# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score
# Read student data
student_data = pd.read_csv("student-data.csv")
#print "Student data read successfully!"

#print  student_data.shape
#df = pd.DataFrame({'vals': [1, 2, 3, 4], 'ids': [u'aball', u'bball', u'cnut', u'fball']})
#counts = student_data['passed'].value_counts()
#print counts['yes']
#print counts['no']

# TODO: Calculate number of students
n_students = student_data.shape[0]

# TODO: Calculate number of features
# -1 : the last column is the target labels
n_features = student_data.shape[1] - 1

# TODO: Calculate passing students
counts = student_data['passed'].value_counts()
n_passed = counts['yes']

# TODO: Calculate failing students
n_failed = counts['no']

# TODO: Calculate graduation rate
grad_rate = n_passed * 100.0  / n_students 

# Print the results
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

# Extract feature columns
feature_cols = list(student_data.columns[:-1])

print feature_cols

# Extract target column 'passed'
target_col = student_data.columns[-2] 

print target_col

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print "\nFeature values:"
print X_all.head()

# TODO: Import any additional functionality you may need here
from sklearn import cross_validation

# TODO: Set the number of training points
num_train = 300

# Set the number of testing points
num_test = X_all.shape[0] - num_train

# TODO: Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_all, y_all, test_size = num_test * 1.0 / X_all.shape[0], random_state = 0)
#X_train = None
#X_test = None
#y_train = None
#y_test = None

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

print X_train.shape

x_1 = X_train.head(100)
X_2 = X_train[:100]

print x_1 == X_2
print X_2.shape
print x_1.shape