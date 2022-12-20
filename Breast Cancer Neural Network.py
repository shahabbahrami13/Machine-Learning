from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
# from sklearn.metrics import r2_score
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from numpy import array
from numpy import var
from numpy import mean

from matplotlib import pyplot as plt


x, y = load_breast_cancer(return_X_y=True)

time = []
acc = []
n_component = 0.95
for times in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=times ** 4)

    minmax = MinMaxScaler()
    x_train = minmax.fit_transform(x_train)
    x_test = minmax.transform(x_test)

    pca = PCA(n_components=n_component)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    # lda = LDA()
    # x_train = lda.fit_transform(x_train, y_train)
    # x_test = lda.transform(x_test)

    model = MLPClassifier(hidden_layer_sizes=len(x_train[0])**2*3,
                          max_iter=600,
                          tol=1e-4)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    # print(times + 1)
    # print(classification_report(y_test, y_pred))
    this_acc = classification_report(y_test, y_pred, output_dict=True, zero_division=1)['accuracy']//0.01
    acc.append(this_acc)
    time.append(times + 1)
# print(y_test)
# print(y_pred)

acc = array(acc)
var = round(var(acc), ndigits=2)
mean = round(mean(acc), ndigits=2)

# print(time)
# print(acc)

plt.figure(figsize=(7, 4))
plt.scatter(time, acc, c='k')
plt.xticks([i + 1 for i in range(len(time))])
plt.yticks([j for j in range(int(acc.min(axis=0)), int(acc.max(axis=0)) + 1, 1)])

plt.title(f"variance={var} / mean={mean} / max_iter=600 / tol=1e-4")
plt.xlabel('time')
plt.ylabel('accuracy [percent]')
plt.savefig('practice4_files/P23.jpg')
plt.show()
