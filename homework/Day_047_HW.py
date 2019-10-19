#作業
import time
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor

# 讀取波士頓房價資料集
boston = datasets.load_boston()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.25, random_state=42)

# 建立模型
clf = GradientBoostingRegressor(random_state=7)

# 先看看使用預設參數得到的結果，約為 8.379 的 MSE
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(metrics.mean_squared_error(y_test, y_pred))

start_time = time.time()
n_estimators = [x for x in range(20, 300, 10)]
max_depth = [x for x in range(1, 15, 2)]
param_range = dict(n_estimators=n_estimators, max_depth=max_depth)

## 建立搜尋物件，放入模型及參數組合字典 (n_jobs=-1 會使用全部 cpu 平行運算)
random_search = RandomizedSearchCV(clf, param_range, scoring="neg_mean_squared_error", random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)

# 開始搜尋最佳參數
random_result = random_search.fit(x_train, y_train)
print("All cores costs %f sec" % (time.time() - start_time))

start_time = time.time()
n_estimators = [x for x in range(20, 300, 10)]
max_depth = [x for x in range(1, 15, 2)]
param_range = dict(n_estimators=n_estimators, max_depth=max_depth)

## 建立搜尋物件，放入模型及參數組合字典 (n_jobs=-1 會使用全部 cpu 平行運算)
random_search = RandomizedSearchCV(clf, param_range, scoring="neg_mean_squared_error", random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=2)

# 開始搜尋最佳參數
random_result = random_search.fit(x_train, y_train)
print("2 cores costs %f sec" % (time.time() - start_time))

print("Best Accuracy: %f using %s" % (random_result.best_score_, random_result.best_params_))

# 使用最佳參數重新建立模型
clf_bestparam = GradientBoostingRegressor(max_depth=random_result.best_params_['max_depth'],
                                           n_estimators=random_result.best_params_['n_estimators'])

# 訓練模型
clf_bestparam.fit(x_train, y_train)

# 預測測試集
y_pred = clf_bestparam.predict(x_test)

# 調整參數後約可降至 8.30 的 MSE
print(metrics.mean_squared_error(y_test, y_pred))