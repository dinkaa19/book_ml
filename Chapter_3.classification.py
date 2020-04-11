#!/usr/bin/env python
# coding: utf-8

# # Классификация (прогозирование классов)
# ## Обучение с учителем
# 

# In[8]:


import numpy as np
def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


# In[9]:


try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
    sort_by_target(mnist) # fetch_openml() returns an unsorted dataset
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
mnist["data"], mnist["target"]


# In[17]:


X,y = mnist['data'], mnist['target']
# 70.000 изображений и 784 признака
# Каждый признак представляет иннтесивность пикселя
# от 0(белый) до 255(черный)


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt


# In[29]:


some_digit = X[36000]
some_digit_image = some_digit.reshape(28,28)


# In[81]:


plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,           interpolation='nearest')
plt.axis('off')
plt.show()


# In[59]:


# Создадим испытальный набор
X_train, X_test, y_train, y_test =     X[:60000],X[60000:],y[:60000],y[60000:]


# In[60]:


# Перетасовка обучающегося набора
shaffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shaffle_index],y_train[shaffle_index]


# ## Двоичный классификатор

# In[120]:


y_train_5 = (y_train == 5) #True для все пятёрок
y_test_5 = (y_test == 5) #False для всех остальных цифр


# Стохастический градиентый спуск (Stochastic Gradient Descent - SGD)

# In[121]:


from sklearn.linear_model import SGDClassifier


# In[122]:


sgd_clf = SGDClassifier( random_state=42)
sgd_clf.fit(X_train, y_train_5)


# In[123]:


sgd_clf.predict([some_digit])


# # Показатели производительности
# ### Насколько хорошо работает классификатор (способы оценки)

# ### Реализация перекрестной проверки 

# In[75]:


from sklearn.model_selection import StratifiedKFold #для стратифицированной выборки
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))


# In[126]:


# accuracy-коэффицинт корректных прогнозов
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')


# In[140]:


from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")


# Ассиметричные наборы -это когда одни классы встречаются чаще других

# ### Матрица неточностей

# Общая идея заключается в том, чтобы подсчитать, сколько раз образцы класса А были отнесены к классу В. Например, для выяснения, сколько раз классифи­катор путал изображения пятерок с тройками, вы могли бы заглянуть в 5-ю строку и 3-й столбец матрицы неточностей. 

# In[141]:


from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)


# In[144]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred)


# Каждая строка в матрице неточностей представляет фактический класс, а каждый столбец - спрогнозированный класс. Первая строка матрицы учи­тывает изображения не пятерок (отрицательньtй класс (negative class)): 53 309 их них были корректно классифицированы как не пятерки (истинно отрицательные классификации (trиe negative - TN)), тогда как оставшиеся 1 270 были ошибочно классифицированы как пятерки (ложноположительные классификации (false positive - FP)). Вторая строка матрицы учитывает изоб­ражения пятерок (положительный класс class)): 947 были ошибоч­но классифицированы как не пятерки (ложноотрицательные классификации negative - FN) ), в то время как оставшиеся 4 474 были корректно клас­сифицированы как пятерки (истинно положительные классификации positive - ТР)).

# In[149]:


# Точность Precision
4474/(4474+1270)


# In[151]:


# Полнота Recall/чувствительность/Доля истинно положительных классификаторов
4474/(4474+947)


# Вывод: изображение представляет петерку, корректно на 77%, кроме того, от обнаруживает только 82% пятерок.

# ### Точность и полнота = Мера F1(F1 score)

# In[155]:


from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred) #точность


# In[157]:


from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)


# Нельзя получить одновременно высокую и полному и точность: увеличение точности снижает полноту и наоборот.

# In[159]:


y_scores = sgd_clf.decision_function([some_digit])
y_scores  #Порог принятия решения


# In[167]:


threshold = 4600
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred


# Поднятие порога снижает полноту

# In[168]:


y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")


# Вычисляем точность и полному

# In[169]:


from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


# In[175]:


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold - Порог", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


# In[176]:


(y_train_pred == (y_scores > 0)).all()


# In[201]:


y_train_pred_90 = (y_scores > 0)


# In[202]:


precision_score(y_train_5, y_train_pred_90)


# In[203]:


recall_score(y_train_5, y_train_pred_90)


# In[200]:


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.show()


# In[ ]:




