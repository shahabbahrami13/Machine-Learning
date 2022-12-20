

# import some needed madule

from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree


breast_cancer = datasets.load_breast_cancer()
x = breast_cancer.data
y = breast_cancer.target
xtrain, xtest, ytrain, ytest = train_test_split(x, y)

scaler = MinMaxScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

dr = PCA(n_components=0.9)
xtrain = dr.fit_transform(xtrain)
xtest = dr.transform(xtest)

dr = LinearDiscriminantAnalysis()
xtrain = dr.fit_transform(xtrain, ytrain)
xtest = dr.transform(xtest)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

model = GaussianNB()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

model = svm.SVC()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

model = tree.DecisionTreeClassifier()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)


print('ytest: ', ytest)
print('ypred: ', ypred)

print('confusion_matrix: ', confusion_matrix(ytest, ypred))

print(classification_report(ytest, ypred))