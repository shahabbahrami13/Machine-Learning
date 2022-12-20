from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
import pandas as pd
from numpy import array
from numpy import var
from numpy import mean

from matplotlib import pyplot as plt


data = pd.read_csv("practice5_files/train.csv", index_col='Id')

y = data['revenue']

x = data.drop(['Open Date', 'revenue'], axis=1)
x = x.values

x = x[:, [0, 1, 2, 3, 4, 6, 8, 10, 13, 14, 21, 22, 23, 25, 26, 28, 33, 34, 35, 36, 37, 38]]
y = y.values
time = []
r2 = []
n_component = 0.95
for times in range(8):
    # times = 4
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=times ** 4)

    'Converting the string to integer'
    converter = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train_categories = converter.fit_transform(x_train[:, 0:4])
    test_categories = converter.transform(x_test[:, 0:4])

    for row in range(len(x_train)):
        for column in range(3):
            x_train[row, column] = train_categories[row, column]

    for row in range(len(x_test)):
        for column in range(3):
            x_test[row, column] = test_categories[row, column]

    'FS & DR'
    pca = PCA(n_components=n_component)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    # print('*'*20)
    # print(x_train[:10, :])

    'Normalizing'
    minmax = MinMaxScaler()
    x_train = minmax.fit_transform(x_train)
    x_test = minmax.transform(x_test)
    # print(x_train[:10, :])

    'Model'
    iteration = 750
    tol = 1e-4
    model = MLPRegressor(hidden_layer_sizes=len(x_train[0])*12, solver='sgd', learning_rate='adaptive',
                         learning_rate_init=5e-4, max_iter=iteration, tol=tol, momentum=0.95)  # , n_iter_no_change=50)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    # print(times + 1)
    # print(classification_report(y_test, y_pred))
    r2.append(r2_score(y_test, y_pred))
    time.append(times + 1)
# print(y_test)
# print(y_pred)

r2 = array(r2)
var = round(var(r2), ndigits=2)
mean = round(mean(r2), ndigits=2)

# print(time)
# print(r2)

plt.figure(figsize=(8, 5.5))
plt.scatter(time, r2, c='k')
plt.xticks([i + 1 for i in range(len(time))])
plt.yticks([round(j, 1) for j in list(r2)])

plt.title(f"variance={var}/mean={mean}/momentum=0.95")
plt.xlabel('time')
plt.ylabel('r2_score')
plt.grid(axis='y')
# plt.savefig('practice5_files/P45.jpg')
plt.show()
