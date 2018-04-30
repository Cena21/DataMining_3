import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess
from pandas import Series, DataFrame
from patsy import dmatrices
from sklearn import datasets, svm
#read the data from training set
data_train = pd.read_csv('./data/train.csv')
#pd = pd.read_csv('./data/gender_submission.csv',low_memory=False)
#data_train.info()
#data_train.describe()
data_train = data_train.drop(['Ticket','Cabin'], axis=1)
# Remove NaN values
data_train = data_train.dropna() 

#将数据可视化
import matplotlib.pyplot as plt
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

#plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind='bar') 
plt.title(u"Survival(1:saved)")  
plt.ylabel(u"PeopleNum")  
plt.show()

#各等级人数分布
#plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"PeopleNum")
plt.title(u"Classes")
plt.show()

#各年龄存活人数分布
#plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"Age")                         # sets the y axis lable
plt.grid(b=True, which='major', axis='y') # formats the grid line style of our graphs
plt.title(u"Age-Survival(1:saved)")
plt.show()

#各等级船舱中年龄分布
#plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')   # plots a kernel desnsity estimate of the subset of the 1st class passanges's age
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"Age")# plots an axis lable
plt.ylabel(u"Denity") 
plt.title(u"Age-Class")
plt.legend((u'1st', u'2nd',u'3rd'),loc='best') # sets our legend for our graph.
plt.show()

#登录港口人数分布       
#plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"Load port PeopleNum")
plt.ylabel(u"PeopleNum")  
plt.show()


#按等级分存活人数分布
fig = plt.figure()
fig.set(alpha=0.2)

S_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
S_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u"Saved":S_1,u"Unsaved":S_0})
df.plot(kind='bar',stacked = True)

plt.title(u"Class&Survival")
plt.xlabel(u"class")
plt.ylabel(u"PeopleNum")
plt.show()

#按性别分存活人数分布
S_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
S_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df_1 =pd.DataFrame({u'Male':S_m,u"Female":S_f})
df_1.plot(kind='bar',stacked = True)
plt.title(u"Sex&Survival")
plt.xlabel(u"Sex")
plt.ylabel(u"PeopleNum")
plt.show()

#1.Logistic Regression
formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp  + C(Embarked)'       
results = {} 
 
y,x = dmatrices(formula, data=df, return_type='dataframe')
model = sm.Logit(y,x)
res = model.fit() 
results['Logit'] = [res, formula]
res.summary()

#2.SVM
 
compared_resuts = ka.predict(test_data, results, 'Logit')
compared_resuts = Series(compared_resuts)  # convert our model to a series for easy output

formula_ml = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp + Parch + C(Embarked)'
plt.figure(figsize=(8,6))

y, x = dmatrices(formula_ml, data=df, return_type='matrix')

feature_1 = 2
feature_2 = 3

X = np.asarray(x)
X = X[:,[feature_1, feature_2]]  


y = np.asarray(y)
y = y.flatten()      

n_sample = len(X)

np.random.seed(0)
order = np.random.permutation(n_sample)

X = X[order]
y = y[order].astype(np.float)

nighty_precent_of_sample = int(.9 * n_sample)
X_train = X[:nighty_precent_of_sample]
y_train = y[:nighty_precent_of_sample]
X_test = X[nighty_precent_of_sample:]
y_test = y[nighty_precent_of_sample:]

types_of_kernels = ['linear', 'rbf', 'poly']

color_map = plt.cm.RdBu_r


for fig_num, kernel in enumerate(types_of_kernels):
    clf = svm.SVC(kernel=kernel, gamma=3)
    clf.fit(X_train, y_train)

    plt.figure(fig_num)
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=color_map)

    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10)
    
    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=color_map)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
               levels=[-.5, 0, .5])

    plt.title(kernel)
    plt.show()

clf = svm.SVC(kernel='poly', gamma=3).fit(X_train, y_train)                                                            
y,x = dmatrices(formula_ml, data=test_data, return_type='dataframe')

res_svm = clf.predict(x.ix[:,[6,3]].dropna()) 
res_svm = DataFrame(res_svm,columns=['Survived'])


#3.Random Fores
import sklearn.ensemble as ske
y, x = dmatrices(formula_ml, data=df, return_type='dataframe')
y = np.asarray(y).ravel()
results_rf = ske.RandomForestClassifier(n_estimators=100).fit(x, y)
score = results_rf.score(x, y)
print "Mean accuracy of Random Forest Predictions on the data was: {0}".format(score)
#Mean accuracy of Random Forest Predictions on the data was: 0.945224719101