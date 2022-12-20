import pandas as pd
from sklearn.metrics import mean_absolute_error

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

# FS & DR
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from genetic_selection import GeneticSelectionCV

# Models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR


# pd.set_option('display.max_columns', 10)

data = pd.read_csv("practice3_files/train.csv", index_col='Id')


"""Preprocessing"""

'Correcting the form of Open Date: 7/7/1999 --> 07/07/1999'
# main_form = data['Open Date'].values
# for row in range(len(main_form)):
#     each_one = main_form[row].split('/')
#     month = each_one[0]
#     day = each_one[1]
#     if len(day) == 1:
#         day = '0' + day
#     if len(month) == 1:
#         month = '0' + month
#     main_form[row] = day + '/' + month + '/' + each_one[2]
# data['Open Date'] = main_form
# data.to_csv("practice3_files/train.csv")
# print(data.head(10))

'''after FS, it has been removed'''
"Changing the type of column 'Open Date'."
# Open_Date = data['Open Date']
# Date = []
# for date in Open_Date:
#     date = datetime.strptime(date, '%m/%d/%Y')
#     Date.append(date)
# data['Open Date'] = Date

'Split train & test data'
y = data['revenue']
# after FS, we also remove 'Open Date'
x = data.drop(['Open Date', 'revenue'], axis=1)
x = x.values
# after FS
x = x[:, [0, 1, 2, 3, 4, 6, 8, 10, 13, 14, 21, 22, 23, 25, 26, 28, 33, 34, 35, 36, 37, 38]]
y = y.values

a = 2
tol = 1e-3
max_iter = 1000
print('\n')
print(f'alpha={a} / tol={tol} / max_iter={max_iter} :')
for i in range(1, 6):
    # i = 1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=i * 10)

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

    '''after FS, it has been removed'''
    'Split dates to separated columns'
    # day = []
    # month = []
    # year = []
    # for row in range(len(x_train)):
    #     day.append(x_train[row, 0].day)
    #     month.append(x_train[row, 0].month)
    #     year.append(x_train[row, 0].year)
    # x_train = delete(x_train, 0, axis=1)
    # x_train = hstack((array(year).reshape(-1, 1), array(month).reshape(-1, 1), array(day).reshape(-1, 1), x_train))
    #
    # day.clear()
    # month.clear()
    # year.clear()
    # for row in range(len(x_test)):
    #     day.append(x_test[row, 0].day)
    #     month.append(x_test[row, 0].month)
    #     year.append(x_test[row, 0].year)
    # x_test = delete(x_test, 0, axis=1)
    # x_test = hstack((array(year).reshape(-1, 1), array(month).reshape(-1, 1), array(day).reshape(-1, 1), x_test))
    # day.clear()
    # month.clear()
    # year.clear()

    'Normalizing'
    minmax = MinMaxScaler()
    x_train = minmax.fit_transform(x_train)
    x_test = minmax.transform(x_test)

    """ DR """
    dr1 = PCA(n_components=0.85)
    x_train = dr1.fit_transform(x_train)
    x_test = dr1.transform(x_test)

    # dr2 = LinearDiscriminantAnalysis()
    # x_train = dr2.fit_transform(x_train, y_train)
    # x_test = dr2.transform(x_test)

    """ Models """
    # model = LinearRegression()
    model = Ridge(alpha=a, tol=tol, max_iter=max_iter)
    # model = Lasso(alpha=3, max_iter=10000, tol=1e-10)
    # model = SVR(kernel='poly', degree=5)

    ''' FS '''
    # if __name__ == "__main__":
    #     selector = GeneticSelectionCV(
    #         model,
    #         cv=5,
    #         verbose=1,
    #         max_features=40,
    #         n_population=60,
    #         crossover_proba=0.5,
    #         mutation_proba=0.2,
    #         n_generations=80,
    #         crossover_independent_proba=0.5,
    #         mutation_independent_proba=0.05,
    #         tournament_size=3,
    #         n_gen_no_change=40,
    #         caching=True,
    #         n_jobs=-1,
    #     )
    #     selector = selector.fit(x_train, y_train)
    #
    #     print(selector.support_)

    """ Fitting the Model """
    model.fit(x_train, y_train)
    predict = model.predict(x_test)

    error_absolut = round(mean_absolute_error(y_test, predict), ndigits=-3)
    print(f'\tin state {i*10} --> {error_absolut}')
print('\n')
print(x_train.shape)
