import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Check unique values in labels
# unique_labels = np.unique(labels)
# print("Unique Labels:", unique_labels)
#
# # Check the data type of labels
# print("Labels Data Type:", labels.dtype)

# integerLabels = labels.astype(int)

print(labels)
# Ensure labels are integers or strings representing classes

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_test, y_predict)

print('{}% of samples were classified correctly!'.format(score * 100))

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
