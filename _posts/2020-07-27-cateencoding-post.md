---
layout: post
title: An Overview of Encoding Techniques
author: Jaeheon Kwon
categories: Ai
tags: [Feature Encoding]
---



카카오 아레나 이후로 데이터 컴피티션에 관심이 많이 생겨서 tabular 데이터를 다루는 법에 대해 공부중입니다. 

처음 이러한 유형의 컴피티션을 접했을 때 가장 와닫지 않은 부분은 바로 categorical 데이터를 다루는 방법이었습니다.

마침 [캐글](https://www.kaggle.com/c/cat-in-the-dat)에 좋은 챌린지가 있어서 1등 노트북을 필사해 봤습니다.



우선 데이터는 아래처럼 생겼습니다.



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>bin_0</th>
      <th>bin_1</th>
      <th>bin_2</th>
      <th>bin_3</th>
      <th>bin_4</th>
      <th>nom_0</th>
      <th>nom_1</th>
      <th>nom_2</th>
      <th>nom_3</th>
      <th>...</th>
      <th>nom_9</th>
      <th>ord_0</th>
      <th>ord_1</th>
      <th>ord_2</th>
      <th>ord_3</th>
      <th>ord_4</th>
      <th>ord_5</th>
      <th>day</th>
      <th>month</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>T</td>
      <td>Y</td>
      <td>Green</td>
      <td>Triangle</td>
      <td>Snake</td>
      <td>Finland</td>
      <td>...</td>
      <td>2f4cb3d51</td>
      <td>2</td>
      <td>Grandmaster</td>
      <td>Cold</td>
      <td>h</td>
      <td>D</td>
      <td>kr</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>T</td>
      <td>Y</td>
      <td>Green</td>
      <td>Trapezoid</td>
      <td>Hamster</td>
      <td>Russia</td>
      <td>...</td>
      <td>f83c56c21</td>
      <td>1</td>
      <td>Grandmaster</td>
      <td>Hot</td>
      <td>a</td>
      <td>A</td>
      <td>bF</td>
      <td>7</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>F</td>
      <td>Y</td>
      <td>Blue</td>
      <td>Trapezoid</td>
      <td>Lion</td>
      <td>Russia</td>
      <td>...</td>
      <td>ae6800dd0</td>
      <td>1</td>
      <td>Expert</td>
      <td>Lava Hot</td>
      <td>h</td>
      <td>R</td>
      <td>Jc</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>F</td>
      <td>Y</td>
      <td>Red</td>
      <td>Trapezoid</td>
      <td>Snake</td>
      <td>Canada</td>
      <td>...</td>
      <td>8270f0d71</td>
      <td>1</td>
      <td>Grandmaster</td>
      <td>Boiling Hot</td>
      <td>i</td>
      <td>D</td>
      <td>kW</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>F</td>
      <td>N</td>
      <td>Red</td>
      <td>Trapezoid</td>
      <td>Lion</td>
      <td>Canada</td>
      <td>...</td>
      <td>b164b72a7</td>
      <td>1</td>
      <td>Grandmaster</td>
      <td>Freezing</td>
      <td>a</td>
      <td>R</td>
      <td>qP</td>
      <td>7</td>
      <td>8</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 300000 entries, 0 to 299999
Data columns (total 25 columns):
id        300000 non-null int64
bin_0     300000 non-null int64
bin_1     300000 non-null int64
bin_2     300000 non-null int64
bin_3     300000 non-null object
bin_4     300000 non-null object
nom_0     300000 non-null object
nom_1     300000 non-null object
nom_2     300000 non-null object
nom_3     300000 non-null object
nom_4     300000 non-null object
nom_5     300000 non-null object
nom_6     300000 non-null object
nom_7     300000 non-null object
nom_8     300000 non-null object
nom_9     300000 non-null object
ord_0     300000 non-null int64
ord_1     300000 non-null object
ord_2     300000 non-null object
ord_3     300000 non-null object
ord_4     300000 non-null object
ord_5     300000 non-null object
day       300000 non-null int64
month     300000 non-null int64
target    300000 non-null int64
dtypes: int64(8), object(17)
memory usage: 57.2+ MB
```



그럼 다양한 방법으로 categorical feature들을 encoding 해보겠습니다.

## 1. Label encoding

---

사실 가장 흔한 방법이고 가장 익숙한 방법입니다.

모든 범주형 데이터를 숫자로 변경합니다. 예를들어 Grandmaster, master, expert가 있을 때, 각각을 1, 2, 3으로 변환하는 것을 뜻합니다.

이를 사용하기 위해 sklearn에서 모듈을 불러옵니다.

```python
from sklearn.preprocessing import LabelEncoder

train=pd.DataFrame()
label=LabelEncoder()
for c in  X.columns:
    if(X[c].dtype=='object'):
        train[c]=label.fit_transform(X[c])
    else:
        train[c]=X[c]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>bin_0</th>
      <th>bin_1</th>
      <th>bin_2</th>
      <th>bin_3</th>
      <th>bin_4</th>
      <th>nom_0</th>
      <th>nom_1</th>
      <th>nom_2</th>
      <th>nom_3</th>
      <th>...</th>
      <th>nom_8</th>
      <th>nom_9</th>
      <th>ord_0</th>
      <th>ord_1</th>
      <th>ord_2</th>
      <th>ord_3</th>
      <th>ord_4</th>
      <th>ord_5</th>
      <th>day</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>...</td>
      <td>1686</td>
      <td>2175</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>7</td>
      <td>3</td>
      <td>136</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>...</td>
      <td>650</td>
      <td>11635</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>93</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>...</td>
      <td>1932</td>
      <td>8078</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>17</td>
      <td>31</td>
      <td>7</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 24 columns</p>

모든 범주형 데이터가 숫자로 바뀐 것을 볼 수 있습니다.

```python
>>>print('train data set has got {} rows and{}columns'.format(train.shape[0],train.shape[1]))

train data set has got 300000 rows and 24 columns
```



각각의 인코딩에 대해 성능을 비교하기위해 심플한 분류 모델을 만듭니다.

### Logistic regression

```python
def logistic(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
    lr=LogisticRegression()
    lr.fit(X_train,y_train)
    y_pre=lr.predict(X_test)
    print('Accuracy : ',accuracy_score(y_test,y_pre))
```

```python
>>>logistic(train,y)
Accuracy :  0.6925333333333333
```



## 2. One Hot Encoding

---

이것도 NLP를 공부해보셨다면 아주 익숙한 표현일 것입니다.

모든 데이터를 고차원 벡터로 변환하는데 여기서 각 차원은 하나의 값만 1이고 나머지는 모두 0입니다.(아주 sparse 하겠죠?)

<img src = "https://py-tonic.github.io/images/label_encoding/1.png">

마찬가지로 sklearn 모듈을 통해 변환할 수 있습니다.(pd.get_dummies 라는 함수로도 가능합니다.)

```python
from sklearn.preprocessing import OneHotEncoder

one=OneHotEncoder()

one.fit(X)
train=one.transform(X)

print('train data set has got {} rows and {} columns'.format(train.shape[0],train.shape[1]))
```

```python
train data set has got 300000 rows and 316461 columns
```

```python
logistic(train,y)
```

```python
Accuracy :  0.7593666666666666
```

보시는 것 처럼 성능이 좋습니다.

하지만 columns수가 24개에서 31만개로 급격히 늘어난 것을 볼 수 있습니다.



## 3. Feature hashing(a.k.a the hashing trick)

---

피처 해싱은 'one-hot-encoding'스타일로 카테고리를 sparse matrix로 표현하지만 차원이 훨씬 낮은 멋진 기술입니다.

피처 해싱에서는 해싱 함수를 카테고리에 적용한 다음 해당 인덱스로 표시합니다.

예를 들어, 'New York'를 나타내기 위해 차원 수를 5로 선택하면, H(New York) mod 5 = 3(예를 들어) 이렇게 계산하면 'New York'의 표현은 (0,0,1,0,0)이 됩니다.

마찬가지로 우리의 친구 sklearn 모듈을 불러옵니다.

```python
from sklearn.feature_extraction import FeatureHasher

X_train_hash=X.copy()
for c in X.columns:
    X_train_hash[c]=X[c].astype('str')      
hashing=FeatureHasher(input_type='string')
train=hashing.transform(X_train_hash.values)

print('train data set has got {} rows and {} columns'.format(train.shape[0],train.shape[1]))
```

```python
train data set has got 300000 rows and 1048576 columns
```

```
logistic(train,y)
```

```python
Accuracy :  0.7512333333333333
```

보시는 것 처럼 one-hot-encoding으로 변환한 데이터만큼의 성능을 보여주지만 차원 수는 훨씬 낮습니다.



## 4. Encoding categories with dataset statistics

---

이제 우리는 모델의 각 피처에 대해 비슷한 범주를 서로 가깝게 배치하는 인코딩을 사용하여 모든 범주에 대한 숫자 표현을 만들어 봅니다.

가장 쉬운 방법은 모든 범주를 데이터 집합에서 나타난 횟수로 바꾸는 것입니다.

이런식으로 뉴욕과 뉴저지가 모두 대도시이면 데이터 세트에서 여러 번 나타날 수 있으며 모델은 이들이 유사하다는 것을 알 수 있습니다.

```python
X_train_stat=X.copy()
for c in X_train_stat.columns:
    if(X_train_stat[c].dtype=='object'):
        X_train_stat[c]=X_train_stat[c].astype('category')
        counts=X_train_stat[c].value_counts()
        counts=counts.sort_index()
        counts=counts.fillna(0)
        counts += np.random.rand(len(counts))/1000
        X_train_stat[c].cat.categories=counts
```



> 사실 random을 통해서 노이즈를 왜 추가해 주는지는 잘 모르겠다.. 
>
> 아시는 분 계시면 댓글 달아주세요!
>
> 

변환 후 아웃풋은 아래와 같다.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>bin_0</th>
      <th>bin_1</th>
      <th>bin_2</th>
      <th>bin_3</th>
      <th>bin_4</th>
      <th>nom_0</th>
      <th>nom_1</th>
      <th>nom_2</th>
      <th>nom_3</th>
      <th>...</th>
      <th>nom_8</th>
      <th>nom_9</th>
      <th>ord_0</th>
      <th>ord_1</th>
      <th>ord_2</th>
      <th>ord_3</th>
      <th>ord_4</th>
      <th>ord_5</th>
      <th>day</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>153535.000826</td>
      <td>191633.000545</td>
      <td>127341.000001</td>
      <td>29855.000948</td>
      <td>45979.000444</td>
      <td>36942.000133</td>
      <td>...</td>
      <td>271.000802</td>
      <td>19.000267</td>
      <td>2</td>
      <td>77428.000323</td>
      <td>33768.000648</td>
      <td>24740.000509</td>
      <td>3974.000977</td>
      <td>506.000990</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>153535.000826</td>
      <td>191633.000545</td>
      <td>127341.000001</td>
      <td>101181.000962</td>
      <td>29487.000190</td>
      <td>101123.000074</td>
      <td>...</td>
      <td>111.000142</td>
      <td>13.000710</td>
      <td>1</td>
      <td>77428.000323</td>
      <td>22227.000155</td>
      <td>35276.000190</td>
      <td>18258.000088</td>
      <td>2603.000907</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>146465.000337</td>
      <td>191633.000545</td>
      <td>96166.000432</td>
      <td>101181.000962</td>
      <td>101295.000088</td>
      <td>101123.000074</td>
      <td>...</td>
      <td>278.000558</td>
      <td>29.000648</td>
      <td>1</td>
      <td>25065.000347</td>
      <td>63908.000426</td>
      <td>24740.000509</td>
      <td>16927.000164</td>
      <td>2572.000012</td>
      <td>7</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 24 columns</p>

```python
logistic(X_train_stat,y)
```

```python
Accuracy :  0.6946166666666667
```



## 5. Encoding cyclic features

---

![](https://miro.medium.com/max/343/1*70cevmU8wNggGJEdLam1lw.png)

날짜, 시간 등과 같이 주기를 갖는 데이터에 대해서는 삼각 함수를 사용하여 데이터를 2차원으로 변환 할 수 있습니다.

```python
X_train_cyclic=X.copy()
columns=['day','month']
for col in columns:
    X_train_cyclic[col+'_sin']=np.sin((2*np.pi*X_train_cyclic[col])/max(X_train_cyclic[col]))
    X_train_cyclic[col+'_cos']=np.cos((2*np.pi*X_train_cyclic[col])/max(X_train_cyclic[col]))
X_train_cyclic=X_train_cyclic.drop(columns,axis=1)

X_train_cyclic[['day_sin','day_cos']].head(3)
```



변환 후 아웃풋은 아래와 같습니다.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>day_sin</th>
      <th>day_cos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.749279e-01</td>
      <td>-0.222521</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.449294e-16</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.449294e-16</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>

주기성을 갖지 않는 나머지 데이터에 대해 one-hot encoding을 적용하여 모델에 넣습니다.

```python
one=OneHotEncoder()

one.fit(X_train_cyclic)
train=one.transform(X_train_cyclic)

print('train data set has got {} rows and {} columns'.format(train.shape[0],train.shape[1]))
```

```python
train data set has got 300000 rows and 316478 columns
```

```
logistic(train,y)
```

```python
Accuracy :  0.75935
```



## 6. Target encoding

---

Target-based 인코딩은 대상을 통한 범주형 변수의 숫자화입니다.

이 방법에서는 범주형 변수를 하나의 새로운 숫자형 변수로 바꾸고, 범주형 변수의 각 범주를 대상의 확률(범주형) 또는 대상의 평균(숫자인 경우)으로 대체합니다.

아래의 예시를 봅시다.



<table style="width : 20%">
    <tr>
    <th>Country</th>
    <th>Target</th>
    </tr>
    <tr>
    <td>India</td>
    <td>1</td>
    </tr>
    <tr>
    <td>China</td>
    <td>0</td>
    </tr>
    <tr>
    <td>India</td>
    <td>0</td>
    </tr>
    <tr>
    <td>China</td>
    <td>1</td>
    </tr>
    </tr>
    <tr>
    <td>India</td>
    <td>1</td>
    </tr>
</table>

인도는 전체 레이블에서 3번 나왔고 실제값은 2번 나왔으므로 인도의 레이블은 2/3 = 0.666입니다.

<table style="width : 20%">
    <tr>
    <th>Country</th>
    <th>Target</th>
    </tr>
    <tr>
    <td>India</td>
    <td>0.66</td>
    </tr>
    <tr>
    <td>China</td>
    <td>0.5</td>
    </tr>
</table>



```python
X_target=df_train.copy()
X_target['day']=X_target['day'].astype('object')
X_target['month']=X_target['month'].astype('object')
for col in X_target.columns:
    if (X_target[col].dtype=='object'):
        target= dict ( X_target.groupby(col)['target'].agg('sum')/X_target.groupby(col)['target'].agg('count'))
        X_target[col]=X_target[col].replace(target).values
```

변환 후 아웃풋은 다음과 같습니다.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>bin_0</th>
      <th>bin_1</th>
      <th>bin_2</th>
      <th>bin_3</th>
      <th>bin_4</th>
      <th>nom_0</th>
      <th>nom_1</th>
      <th>nom_2</th>
      <th>nom_3</th>
      <th>...</th>
      <th>nom_9</th>
      <th>ord_0</th>
      <th>ord_1</th>
      <th>ord_2</th>
      <th>ord_3</th>
      <th>ord_4</th>
      <th>ord_5</th>
      <th>day</th>
      <th>month</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.302537</td>
      <td>0.290107</td>
      <td>0.327145</td>
      <td>0.360978</td>
      <td>0.307162</td>
      <td>0.242813</td>
      <td>...</td>
      <td>0.368421</td>
      <td>2</td>
      <td>0.403885</td>
      <td>0.257877</td>
      <td>0.306993</td>
      <td>0.208354</td>
      <td>0.401186</td>
      <td>0.322048</td>
      <td>0.244432</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.302537</td>
      <td>0.290107</td>
      <td>0.327145</td>
      <td>0.290054</td>
      <td>0.359209</td>
      <td>0.289954</td>
      <td>...</td>
      <td>0.076923</td>
      <td>1</td>
      <td>0.403885</td>
      <td>0.326315</td>
      <td>0.206599</td>
      <td>0.186877</td>
      <td>0.303880</td>
      <td>0.340292</td>
      <td>0.327496</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.309384</td>
      <td>0.290107</td>
      <td>0.241790</td>
      <td>0.290054</td>
      <td>0.293085</td>
      <td>0.289954</td>
      <td>...</td>
      <td>0.172414</td>
      <td>1</td>
      <td>0.317175</td>
      <td>0.403126</td>
      <td>0.306993</td>
      <td>0.351864</td>
      <td>0.206843</td>
      <td>0.340292</td>
      <td>0.244432</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.309384</td>
      <td>0.290107</td>
      <td>0.351052</td>
      <td>0.290054</td>
      <td>0.307162</td>
      <td>0.339793</td>
      <td>...</td>
      <td>0.227273</td>
      <td>1</td>
      <td>0.403885</td>
      <td>0.360961</td>
      <td>0.330148</td>
      <td>0.208354</td>
      <td>0.355985</td>
      <td>0.322048</td>
      <td>0.255729</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 25 columns</p>

```python
logistic(X_target.drop('target',axis=1),y)
```

```python
Accuracy :  0.6946166666666667
```



## Summary

---

전체적으로 one-hot 기반이 성능이 좋은 것을 볼 수 있습니다.

저는 딥러닝을 공부하고 -> tabular데이터(ML계열)로 넘어온 케이스기 때문에 왜 인코딩을 저렇게하지? Word2Vec처럼 모델을 이용해서 object타입간의 유사도를 통한 인코딩이 훨씬 성능이 좋지않나? 라고 생각을 했습니다.

그러나 모델의 피처가 클 경우 저런 방법은 현실적으로 너무 비용이 비싸다는 문제가 있을 것 같습니다.(언젠가 저렇게 모델을 통해 피처를 인코딩하거나 전처리하는 방법을 보게되면 그 때 포스팅 하겠습니다.)



<table style="width : 50%">
    <tr>
    <th>Encoding</th>
    <th>Score</th>
    <th>Wall time</th>
    </tr>
    <tr>
    <td>Label Encoding</td>
    <td>0.692</td>
    <td> 973 ms</td>
    </tr>
    <tr>
    <td>OnHotEncoder</td>
    <td>0.759</td>
    <td>1.84 s</td>
    </tr>
    <tr>
    <td>Feature Hashing</td>
    <td>0.751</td>
    <td>4.96 s</td>
    </tr>
    <tr>
    <td>Dataset statistic encoding</td>
    <td>0.694</td>
    <td>894 ms</td>
    </tr>
    </tr>
    <tr>
    <td>Cyclic + OnHotEncoding</td>
    <td>0.759</td>
    <td>431 ms</td>
    </tr>
    </tr>
    <tr>
    <td>Target encoding</td>
    <td>0.694</td>
    <td>2min 5s</td>
    </tr>


