import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import svm, datasets,model_selection
from sklearn.metrics import accuracy_score
# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features. We could
 # avoid this ugly slicing by using a two-dim dataset
y = iris.target
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
x_train,x_test,y_train,y_test=model_selection.train_test_split(X,y,random_state=1,test_size=0.3)
C = 1.0

svc1 = svm.SVC(kernel='linear', C=1,gamma='auto').fit(X, y)
a1=svc1.score(x_train,y_train)
b1=svc1.score(x_test,y_test)

svc2 = svm.SVC(kernel='poly', C=1,gamma='auto').fit(X, y)
a2=svc2.score(x_train,y_train)
b2=svc2.score(x_test,y_test)

svc3 = svm.SVC(kernel='rbf', C=1,gamma='auto').fit(X, y)
a3=svc3.score(x_train,y_train)
b3=svc3.score(x_test,y_test)


x_tmp=[0,1,2]
y_train_tmp=[a1,a2,a3]
y_test_tmp=[b1,b2,b3]
#设置尺寸和颜色
plt.figure(facecolor='w',figsize=(12,6))
#subplot是用来分割画布，分割成一行两列，用第一列的来画图
a=plt.subplot(121)
#绘图
plt.plot(x_tmp,y_train_tmp,'r-',lw=2,label=u'Training set accuracy')
plt.plot(x_tmp,y_test_tmp,'g-',lw=2,label=u'Testing set accuracy')
#设置两个图形的解释
plt.legend(loc='lower left')
plt.title(u'Model prediction accuracy', fontsize=13)
plt.xticks(x_tmp, [u'linear-SVM', u'poly-SVM', u'rbf-SVM'], rotation=0)
#开启网格线
plt.grid(b=True)
plt.show()


# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
plt.figure(figsize=(10,2))
plt.subplots_adjust(wspace=0.2)
plt.subplot(1, 3, 1)
Z = svc1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cm_dark=plt.get_cmap('Purples')
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_dark)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('C = 1')

plt.subplot(1, 3, 2)
Z = svc2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cm_dark=plt.get_cmap('Purples')
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_dark)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('C = 10')

plt.subplot(1, 3, 3)
Z = svc3.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cm_dark=plt.get_cmap('Purples')
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_dark)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('C = 100')
plt.show()
