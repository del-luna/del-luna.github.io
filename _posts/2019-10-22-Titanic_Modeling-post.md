---
layout: post
title: Titanic Modeling
author: Jaeheon Kwon
categories: Kaggle
tags: [model,titanic]
---

# Titanic Modeling

[Titanic EDA](https://jaeheondev.github.io/Titanic_EDA-post/)

위의 EDA편을 읽고 오시는 것을 추천드립니다.  

이번 시간엔 Modeling에 대해서 다룹니다.  

다양한 모델들과 교차검증, 앙상블 등 다양한 머신러닝 기법에 대해 배웁니다.  


## Predictive Modeling

우리는 EDA를 통해 통찰력을 얻었지만 아직 승객의 생존 여부를 완벽히 예측할 수 없습니다.  
이제 승객의 생존여부를 분류하기위해 어떤 알고리즘을 사용할지 여부를 예측해봅시다.  
우리는 아래의 모델들을 사용합니다.  
1)Logistic Regression

2)Support Vector Machines(Linear and radial)

3)Random Forest

4)K-Nearest Neighbours

5)Naive Bayes

6)Decision Tree


튜토리얼이기 때문에 모델에대한 간략한 설명 정리해봤습니다.  
[Click Link]()를 참고해주세요


```python
#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
```


```python
train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Survived'])
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]
X=data[data.columns[1:]]
Y=data['Survived']
```

#### Radial Support Vector Machines(rbf-SVM)


```python
model=svm.SVC(kernel='rbf',C=1,gamma=0.1)
model.fit(train_X,train_Y)
prediction1=model.predict(test_X)
print('Accuracy for rbf SVM is ',metrics.accuracy_score(prediction1,test_Y))
```

    Accuracy for rbf SVM is  0.835820895522388


#### Linear Support Vector Machine(linear-SVM)


```python
model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)
model.fit(train_X,train_Y)
prediction2=model.predict(test_X)
print('Accuracy for linear SVM is',metrics.accuracy_score(prediction2,test_Y))
```

    Accuracy for linear SVM is 0.8171641791044776


#### Logistic Regression


```python
model = LogisticRegression()
model.fit(train_X,train_Y)
prediction3=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,test_Y))
```

    The accuracy of the Logistic Regression is 0.8171641791044776


#### Decision Tree


```python
model=DecisionTreeClassifier()
model.fit(train_X,train_Y)
prediction4=model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction4,test_Y))
```

    The accuracy of the Decision Tree is 0.7985074626865671


#### K-Nearest Neighbours(KNN)


```python
model=KNeighborsClassifier() 
model.fit(train_X,train_Y)
prediction5=model.predict(test_X)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction5,test_Y))
```

    The accuracy of the KNN is 0.832089552238806


KNN 모델의 파라미터 K를 변경해가면서 정확도를 측정해봅시다.


```python
a_index=list(range(1,11))
a=pd.Series()
x=[0,1,2,3,4,5,6,7,8,9,10]
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i) 
    model.fit(train_X,train_Y)
    prediction=model.predict(test_X)
    a=a.append(pd.Series(metrics.accuracy_score(prediction,test_Y)))
plt.plot(a_index, a)
plt.xticks(x)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()
print('Accuracies for different values of n are:',a.values,'with the max value as ',a.values.max())
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_98_0.png">


    Accuracies for different values of n are: [0.75746269 0.79104478 0.80970149 0.80223881 0.83208955 0.81716418
     0.82835821 0.83208955 0.8358209  0.83208955] with the max value as  0.835820895522388


#### Gaussian Naive Bayes


```python
model=GaussianNB()
model.fit(train_X,train_Y)
prediction6=model.predict(test_X)
print('The accuracy of the NaiveBayes is',metrics.accuracy_score(prediction6,test_Y))
```

    The accuracy of the NaiveBayes is 0.8134328358208955


#### Random Forests


```python
model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_Y)
prediction7=model.predict(test_X)
print('The accuracy of the Random Forests is',metrics.accuracy_score(prediction7,test_Y))
```

    The accuracy of the Random Forests is 0.8097014925373134


분류기의 정확도가 90%라고 했을 때  
train or test 데이터가 변경되면 정확도도 변경됩니다.  
이를 모델 분산이라고 부릅니다.

이를 극복하고 일반화 된 모델을 얻기 위해 교차 검증을 사용합니다.

## Cross Validation

데이터가 불균형합니다.  
즉, class1의 인스턴스는 많지만 다른 클래스의 인스턴스는 적을 수 있습니다.  
따라서 데이터 셋의 모든 인스턴스에서 알고리즘을 학습하고 테스트해야 합니다.  
그런 다음 데이터 셋에 대한 모든 정확도의 평균을 취할 수 있습니다.  

K-Fold 교차검증은 데이터 집합을 K개의 subset으로 나눕니다.  
테스트를 위한 1개를 제외한 K-1개의 subset에 대해 알고리즘을 학습합니다.  
이렇게 하는 이유는 위에서 설명한 것 처럼 알고리즘이 일부 학습 데이터셋에 적합하지 않을 수 있으며 때로는 다른 학습 데이터에 적합할 수도 있기 때문입니다.  
따라서 교차검증을 통해 일반화된 모델을 얻을 수 있습니다.  

교차검증에 대한 설명을 간략히 하자면  
교차검증이란 통계학에서 모델을 평가하는 방법입니다.  

모델을 평가하기 위해 기본적인 방법은 바로 트레이닝셋과 테스트셋을 분리하여 트레이닝셋으로 모델의 계수를 추정하고 테스트셋으로 성능을 평가하는 것인데요.  

교차검증은 이러한 기본적인 작업의 문제점을 보완하기 위해서 쓰입니다. 이 문제점이란 바로 데이터셋의 크기가 작은 경우 테스트셋에 대한 성능 평가의 신뢰성이 떨어지게 된다는 것입니다. 테스트셋을 어떻게 잡느냐에 따라 성능이 아주 상이하게 나온다면, 우연에 의한 효과로 인해 모델 평가 지표에 편향이 생기게 되겠죠.  
[source & description](https://3months.tistory.com/321)


```python
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[]
std=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]
for i in models:
    model = i
    cv_result = cross_val_score(model,X,Y, cv = kfold,scoring = "accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
new_models_dataframe2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CV Mean</th>
      <th>Std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Linear Svm</th>
      <td>0.793471</td>
      <td>0.047797</td>
    </tr>
    <tr>
      <th>Radial Svm</th>
      <td>0.828290</td>
      <td>0.034427</td>
    </tr>
    <tr>
      <th>Logistic Regression</th>
      <td>0.805843</td>
      <td>0.021861</td>
    </tr>
    <tr>
      <th>KNN</th>
      <td>0.813783</td>
      <td>0.041210</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>0.805880</td>
      <td>0.031465</td>
    </tr>
    <tr>
      <th>Naive Bayes</th>
      <td>0.801386</td>
      <td>0.028999</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>0.812622</td>
      <td>0.033735</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.subplots(figsize=(12,6))
box=pd.DataFrame(accuracy,index=[classifiers])
box.T.boxplot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1eef1264c50>



<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_107_1.png">



```python
new_models_dataframe2['CV Mean'].plot.barh(width=0.8)
plt.title('Average CV Mean Accuracy')
fig=plt.gcf()
fig.set_size_inches(8,5)
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_108_0.png">


위에서 말한 불균형으로 인해 분류 정확도가 잘못 될 수 있습니다.  
모델이 어디서 잘못되었는지 또는 모델이 잘못 예측한클래스를 보여주는 Confusion Matrix를 이용하여 요약 된 결과를 얻을 수 있습니다,

### Confusion Matrix


```python
f,ax=plt.subplots(3,3,figsize=(12,10))
y_pred = cross_val_predict(svm.SVC(kernel='rbf'),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')
ax[0,0].set_title('Matrix for rbf-SVM')
y_pred = cross_val_predict(svm.SVC(kernel='linear'),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,1],annot=True,fmt='2.0f')
ax[0,1].set_title('Matrix for Linear-SVM')
y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,2],annot=True,fmt='2.0f')
ax[0,2].set_title('Matrix for KNN')
y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,0],annot=True,fmt='2.0f')
ax[1,0].set_title('Matrix for Random-Forests')
y_pred = cross_val_predict(LogisticRegression(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,1],annot=True,fmt='2.0f')
ax[1,1].set_title('Matrix for Logistic Regression')
y_pred = cross_val_predict(DecisionTreeClassifier(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,2],annot=True,fmt='2.0f')
ax[1,2].set_title('Matrix for Decision Tree')
y_pred = cross_val_predict(GaussianNB(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[2,0],annot=True,fmt='2.0f')
ax[2,0].set_title('Matrix for Naive Bayes')
plt.subplots_adjust(hspace=0.2,wspace=0.2)
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_110_0.png">


Matrix를 잘 살펴보면 RBF-SVM은 죽은 승객을 정확하게 예측할 가능성이 높고,  
Navie Bayes는 생존 승객을 정확히 예측할 가능성이 높습니다.

### Hyper-Parameters Tuning
머신 러닝 모델은 블랙 박스와 같습니다.  
이 블랙 박스에는 몇 가지 기본 매개 변수 값이 있으며, 더 나은 모델을 얻기 위해 튜닝하거나 변경할 수 있습니다.   
SVM 모델의 C 및 감마 와 같은 매개변수를 뜻합니다.     
우리는 가장 정확도가 높은 두가지 분류기인 SVM과 RandomForests의 하이퍼파라미터를 조정합니다

#### SVM


```python
from sklearn.model_selection import GridSearchCV
C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['rbf','linear']
hyper={'kernel':kernel,'C':C,'gamma':gamma}
gd=GridSearchCV(estimator=svm.SVC(),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)
```

    Fitting 3 folds for each of 240 candidates, totalling 720 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    0.8282828282828283
    SVC(C=0.5, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)


    [Parallel(n_jobs=1)]: Done 720 out of 720 | elapsed:   13.9s finished


#### Random Forest


```python
n_estimators=range(100,1000,100)
hyper={'n_estimators':n_estimators}
gd=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)
```

    Fitting 3 folds for each of 9 candidates, totalling 27 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done  27 out of  27 | elapsed:   21.6s finished


    0.8170594837261503
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=900,
                           n_jobs=None, oob_score=False, random_state=0, verbose=0,
                           warm_start=False)


RBF-SVM의 최고 점수는(C=0.05,gamma=0.1)에서 82.82이고,  
RF의 최고 점수는 (n_etimators=900)에서 81.8이다.

## Ensembling
1. Voting Classifier
2. Bagging
3. Boosting

### Voting Classifier
정확도가 80%인 분류기 여러개를 훈련시켰다고 가정합시다. 더 좋은 분류기를 만드는 간단한 방법은 각 분류기의 예측을 모아서 가장 많이 선택된 클래스를 예측하는 것입니다. 이렇게 다수결 투표로 정해지는 분류기를 Hard voting 라고 합니다.

만약 모든 분류기가 클래스의 확률을 예측할 수 있으면 개별 분류기의 예측을 평균 내어 확률이 가장 높은 클래스를 예측할 수 있습니다. 이를 Soft voting 라고 합니다.

투표기반 분류기에서 중요한 가정은 분류기가 모두 독립이고 오차에 상관관계가 없어야 한다는 점입니다.

같은 데이터로 훈련시키게 되면 분류기가 같은 종류의 오차를 만들기 쉽기 때문에 잘못된 클래스가 다수인 경우가 많고 앙상블의 정확도가 낮아집니다.


```python
from sklearn.ensemble import VotingClassifier
ensemble_lin_rbf=VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=10)),
                                              ('RBF',svm.SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),
                                              ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),
                                              ('LR',LogisticRegression(C=0.05)),
                                              ('DT',DecisionTreeClassifier(random_state=0)),
                                              ('NB',GaussianNB()),
                                              ('svm',svm.SVC(kernel='linear',probability=True))
                                             ], 
                       voting='soft').fit(train_X,train_Y)
print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(test_X,test_Y))
cross=cross_val_score(ensemble_lin_rbf,X,Y, cv = 10,scoring = "accuracy")
print('The cross validated score is',cross.mean())
```

    The accuracy for ensembled model is: 0.8246268656716418
    The cross validated score is 0.8226549199863806


### Bagging
앞서 말햇듯이 다양한 분류기를 만드는 한 가지 방법은 각기다른 훈련 알고리즘을 사용하는 것입니다.

또 다른 방법은 같은 알고리즘을 사용하지만 training set의 서브셋을 무작위로 구성하여 분류기를 각기 다르게 학습 시키는 것입니다.

training set에서 중복을 허용하여 샘플링하는 방식을 bagging이라 합니다.  
bagging은 분산이 높은 모델에 적합합니다.(Ex : 결정트리, RF)  
또한 우리는 작은 값의 neighbours로 KNN을 사용할 수 있습니다.

#### Bagged KNN


```python
from sklearn.ensemble import BaggingClassifier
model=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators=700)
model.fit(train_X,train_Y)
prediction=model.predict(test_X)
print('The accuracy for bagged KNN is:',metrics.accuracy_score(prediction,test_Y))
result=cross_val_score(model,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for bagged KNN is:',result.mean())
```

    The accuracy for bagged KNN is: 0.835820895522388
    The cross validated score for bagged KNN is: 0.8148893428668709


#### Bagged DecisionTree


```python
model=BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=0,n_estimators=100)
model.fit(train_X,train_Y)
prediction=model.predict(test_X)
print('The accuracy for bagged Decision Tree is:',metrics.accuracy_score(prediction,test_Y))
result=cross_val_score(model,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for bagged Decision Tree is:',result.mean())
```

    The accuracy for bagged Decision Tree is: 0.8246268656716418
    The cross validated score for bagged Decision Tree is: 0.8204826353421859


### Boosting
부스팅은 분류기의 순차적 학습을 사용하는 앙상블 기법입니다.  
약한 모델을 순차적으로 강화합니다.  

모델은 먼저 전체 데이터 셋에 대해 학습됩니다.  
이제 모델이 잘못 되었을 때 인스턴스를 얻습니다.  
이제 다음 반복에서 잘못 예측된 인스턴스에 더 집중하거나 더 많은 가중치를 부여합니다.  
따라서 잘못된 인스턴스를 올바르게 예측하려고 시도합니다.  
이제 이 프로세스는 계속 진행되고 정확도의 한계에 도달할 때까지 새로운 분류기가 모델에 추가됩니다.

#### AdaBoost(Adaptive Boosting)


```python
from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)
result=cross_val_score(ada,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for AdaBoost is:',result.mean())
```

    The cross validated score for AdaBoost is: 0.8249526160481218


#### Stochastic Gradient Boosting


```python
from sklearn.ensemble import GradientBoostingClassifier
grad=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)
result=cross_val_score(grad,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for Gradient Boosting is:',result.mean())
```

    The cross validated score for Gradient Boosting is: 0.8182862331176939


#### XGBoost


```python
import xgboost as xg
xgboost=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
result=cross_val_score(xgboost,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for XGBoost is:',result.mean())
```

    The cross validated score for XGBoost is: 0.8104710021563954


Adaboost에서 정확도가 가장 높으므로 Hyperparameter Tuning을 해보겠습니다.


```python
n_estimators=list(range(100,1100,100))
learn_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
hyper={'n_estimators':n_estimators,'learning_rate':learn_rate}
gd=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)
```

    Fitting 3 folds for each of 120 candidates, totalling 360 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 360 out of 360 | elapsed:  5.6min finished


    0.8316498316498316
    AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.05,
                       n_estimators=200, random_state=None)


### Confusion Matrix for the Best Model


```python
ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.05)
result=cross_val_predict(ada,X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,result),cmap='winter',annot=True,fmt='2.0f')
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_137_0.png">


### Feature Importance


```python
f,ax=plt.subplots(2,2,figsize=(15,12))
model=RandomForestClassifier(n_estimators=500,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,0])
ax[0,0].set_title('Feature Importance in Random Forests')
model=AdaBoostClassifier(n_estimators=200,learning_rate=0.05,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,1],color='#ddff11')
ax[0,1].set_title('Feature Importance in AdaBoost')
model=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,0],cmap='RdYlGn_r')
ax[1,0].set_title('Feature Importance in Gradient Boosting')
model=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,1],color='#FD0F00')
ax[1,1].set_title('Feature Importance in XgBoost')
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_139_0.png">

