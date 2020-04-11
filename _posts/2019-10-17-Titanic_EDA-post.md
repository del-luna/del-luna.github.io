---
layout: post
title: Titanic EDA
author: Jaeheon Kwon
categories: Kaggle
tags: [eda,titanic]
---

# Titanic EDA

[source](https://www.kaggle.com/ash316/eda-to-prediction-dietanic)

위 출처의 내용에 조금 설명을 덧붙인 글입니다.  

EDA부분에서 초보위주로 잘 정리된 글이라 가져왔습니다.  

나중엔 시각화 도구(matplotlib,seaborn)와 Pandas를 같이 사용하는 부분을 따로 포스팅 할 예정입니다.  

```python
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
```


```python
data=pd.read_csv('train.csv')
```


```python
data.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.isnull().sum() #checking for total null values
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64



## Survived(Label)


```python
f,ax = plt.subplots(1,2,figsize=(18,8))
data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=data,ax=ax[1])
ax[1].set_title('Survived')
ax[1].set_ylabel('')
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_5_0.png">


성별과 생존자 수로 group화,  
특정 column (여기서는 생존자수)의 수를 보여줍니다.


```python
data.groupby(['Sex','Survived'])['Survived'].count()
```




    Sex     Survived
    female  0            81
            1           233
    male    0           468
            1           109
    Name: Survived, dtype: int64




```python
f,ax = plt.subplots(1,2,figsize=(18,8))
data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=data,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_8_0.png">


첫 번째 그래프는  
Sex column을 기준으로 데이터를 그룹화 한 다음  
Sex,Survived column의 평균값을 계산합니다.  
countplot의 hue 값은 'Sex' column과 비교할 값입니다.

위 그래프를 보면  
배에 있는 남자의 수는 여자의 수보다 훨씬 많지만, 여성 생존자 수는 남성의 거의 두배입니다.  
선박의 여성 생존율은 약 **<u>75%</u>**이지만, 남성의 생존율은 약 **<u>18-19%</u>**입니다.

## Pclass

crosstab은 table을 생성합니다.  
**<u>Pclass</u>**를 기준으로 **<u>Survived</u>**의 빈도를 구합니다.


```python
pd.crosstab(data.Pclass,data.Survived,margins=True).style.background_gradient(cmap='summer_r')
```




<style  type="text/css" >
    #T_771f10d2_f0ac_11e9_ab55_bc838524d564row0_col0 {
            background-color:  #ffff66;
        }    #T_771f10d2_f0ac_11e9_ab55_bc838524d564row0_col1 {
            background-color:  #cee666;
        }    #T_771f10d2_f0ac_11e9_ab55_bc838524d564row0_col2 {
            background-color:  #f4fa66;
        }    #T_771f10d2_f0ac_11e9_ab55_bc838524d564row1_col0 {
            background-color:  #f6fa66;
        }    #T_771f10d2_f0ac_11e9_ab55_bc838524d564row1_col1 {
            background-color:  #ffff66;
        }    #T_771f10d2_f0ac_11e9_ab55_bc838524d564row1_col2 {
            background-color:  #ffff66;
        }    #T_771f10d2_f0ac_11e9_ab55_bc838524d564row2_col0 {
            background-color:  #60b066;
        }    #T_771f10d2_f0ac_11e9_ab55_bc838524d564row2_col1 {
            background-color:  #dfef66;
        }    #T_771f10d2_f0ac_11e9_ab55_bc838524d564row2_col2 {
            background-color:  #90c866;
        }    #T_771f10d2_f0ac_11e9_ab55_bc838524d564row3_col0 {
            background-color:  #008066;
        }    #T_771f10d2_f0ac_11e9_ab55_bc838524d564row3_col1 {
            background-color:  #008066;
        }    #T_771f10d2_f0ac_11e9_ab55_bc838524d564row3_col2 {
            background-color:  #008066;
        }</style>  
<table id="T_771f10d2_f0ac_11e9_ab55_bc838524d564" > 
<thead>    <tr> 
        <th class="index_name level0" >Survived</th> 
        <th class="col_heading level0 col0" >0</th> 
        <th class="col_heading level0 col1" >1</th> 
        <th class="col_heading level0 col2" >All</th> 
    </tr>    <tr> 
        <th class="index_name level0" >Pclass</th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_771f10d2_f0ac_11e9_ab55_bc838524d564level0_row0" class="row_heading level0 row0" >1</th> 
        <td id="T_771f10d2_f0ac_11e9_ab55_bc838524d564row0_col0" class="data row0 col0" >80</td> 
        <td id="T_771f10d2_f0ac_11e9_ab55_bc838524d564row0_col1" class="data row0 col1" >136</td> 
        <td id="T_771f10d2_f0ac_11e9_ab55_bc838524d564row0_col2" class="data row0 col2" >216</td> 
    </tr>    <tr> 
        <th id="T_771f10d2_f0ac_11e9_ab55_bc838524d564level0_row1" class="row_heading level0 row1" >2</th> 
        <td id="T_771f10d2_f0ac_11e9_ab55_bc838524d564row1_col0" class="data row1 col0" >97</td> 
        <td id="T_771f10d2_f0ac_11e9_ab55_bc838524d564row1_col1" class="data row1 col1" >87</td> 
        <td id="T_771f10d2_f0ac_11e9_ab55_bc838524d564row1_col2" class="data row1 col2" >184</td> 
    </tr>    <tr> 
        <th id="T_771f10d2_f0ac_11e9_ab55_bc838524d564level0_row2" class="row_heading level0 row2" >3</th> 
        <td id="T_771f10d2_f0ac_11e9_ab55_bc838524d564row2_col0" class="data row2 col0" >372</td> 
        <td id="T_771f10d2_f0ac_11e9_ab55_bc838524d564row2_col1" class="data row2 col1" >119</td> 
        <td id="T_771f10d2_f0ac_11e9_ab55_bc838524d564row2_col2" class="data row2 col2" >491</td> 
    </tr>    <tr> 
        <th id="T_771f10d2_f0ac_11e9_ab55_bc838524d564level0_row3" class="row_heading level0 row3" >All</th> 
        <td id="T_771f10d2_f0ac_11e9_ab55_bc838524d564row3_col0" class="data row3 col0" >549</td> 
        <td id="T_771f10d2_f0ac_11e9_ab55_bc838524d564row3_col1" class="data row3 col1" >342</td> 
        <td id="T_771f10d2_f0ac_11e9_ab55_bc838524d564row3_col2" class="data row3 col2" >891</td> 
    </tr></tbody> 
</table> 




```python
f,ax=plt.subplots(1,2,figsize=(18,8))
data['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Number Of Passengers By Pclass')
ax[0].set_ylabel('Count')
sns.countplot('Pclass',hue='Survived',data=data,ax=ax[1])
ax[1].set_title('Pclass:Survived vs Dead')
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_14_0.png">


위의 두번째 도표를 보면 알겠지만, Pclass가 높으면 생존률이 높은 것을 알 수 있습니다.  
우리는 돈과 성별이 생존에 중요한 feature임을 확인했습니다.  
사실이러한 예측은 그 시대를 기준으로 생각해보거나 영화를 봤다면 당연히 알 수 있는 feature들 입니다.


```python
pd.crosstab([data.Sex,data.Survived],data.Pclass,margins=True).style.background_gradient(cmap='summer_r')
```




<style  type="text/css" >
    #T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row0_col0 {
            background-color:  #ffff66;
        }    #T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row0_col1 {
            background-color:  #ffff66;
        }    #T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row0_col2 {
            background-color:  #f1f866;
        }    #T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row0_col3 {
            background-color:  #ffff66;
        }    #T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row1_col0 {
            background-color:  #96cb66;
        }    #T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row1_col1 {
            background-color:  #a3d166;
        }    #T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row1_col2 {
            background-color:  #f1f866;
        }    #T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row1_col3 {
            background-color:  #cfe766;
        }    #T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row2_col0 {
            background-color:  #a7d366;
        }    #T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row2_col1 {
            background-color:  #85c266;
        }    #T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row2_col2 {
            background-color:  #6eb666;
        }    #T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row2_col3 {
            background-color:  #85c266;
        }    #T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row3_col0 {
            background-color:  #cde666;
        }    #T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row3_col1 {
            background-color:  #f0f866;
        }    #T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row3_col2 {
            background-color:  #ffff66;
        }    #T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row3_col3 {
            background-color:  #f7fb66;
        }    #T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row4_col0 {
            background-color:  #008066;
        }    #T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row4_col1 {
            background-color:  #008066;
        }    #T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row4_col2 {
            background-color:  #008066;
        }    #T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row4_col3 {
            background-color:  #008066;
        }</style>  
<table id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564" > 
<thead>    <tr> 
        <th class="blank" ></th> 
        <th class="index_name level0" >Pclass</th> 
        <th class="col_heading level0 col0" >1</th> 
        <th class="col_heading level0 col1" >2</th> 
        <th class="col_heading level0 col2" >3</th> 
        <th class="col_heading level0 col3" >All</th> 
    </tr>    <tr> 
        <th class="index_name level0" >Sex</th> 
        <th class="index_name level1" >Survived</th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564level0_row0" class="row_heading level0 row0" >female</th> 
        <th id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564level1_row0" class="row_heading level1 row0" >0</th> 
        <td id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row0_col0" class="data row0 col0" >3</td> 
        <td id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row0_col1" class="data row0 col1" >6</td> 
        <td id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row0_col2" class="data row0 col2" >72</td> 
        <td id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row0_col3" class="data row0 col3" >81</td> 
    </tr>    <tr>
        <th id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564level0_row0" class="row_heading level0 row0" >female</th>
        <th id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564level1_row1" class="row_heading level1 row1" >1</th> 
        <td id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row1_col0" class="data row1 col0" >91</td> 
        <td id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row1_col1" class="data row1 col1" >70</td> 
        <td id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row1_col2" class="data row1 col2" >72</td> 
        <td id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row1_col3" class="data row1 col3" >233</td> 
    </tr>    <tr> 
        <th id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564level0_row2" class="row_heading level0 row2" >male</th> 
        <th id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564level1_row2" class="row_heading level1 row2" >0</th> 
        <td id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row2_col0" class="data row2 col0" >77</td> 
        <td id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row2_col1" class="data row2 col1" >91</td> 
        <td id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row2_col2" class="data row2 col2" >300</td> 
        <td id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row2_col3" class="data row2 col3" >468</td> 
    </tr>    <tr>
        <th id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564level0_row2" class="row_heading level0 row2" >male</th>
        <th id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564level1_row3" class="row_heading level1 row3" >1</th> 
        <td id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row3_col0" class="data row3 col0" >45</td> 
        <td id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row3_col1" class="data row3 col1" >17</td> 
        <td id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row3_col2" class="data row3 col2" >47</td> 
        <td id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row3_col3" class="data row3 col3" >109</td> 
    </tr>    <tr> 
        <th id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564level0_row4" class="row_heading level0 row4" >All</th> 
        <th id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564level1_row4" class="row_heading level1 row4" ></th> 
        <td id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row4_col0" class="data row4 col0" >216</td> 
        <td id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row4_col1" class="data row4 col1" >184</td> 
        <td id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row4_col2" class="data row4 col2" >491</td> 
        <td id="T_8aeafa5c_f0ac_11e9_aaac_bc838524d564row4_col3" class="data row4 col3" >891</td> 
    </tr></tbody> 
</table> 



Cateogorical feature이기 때문에 factorplot을 사용합니다.  
아래의 factorplot은 Pclass별 Survived 비율을 Sex로 비교해서 나타내줍니다.


```python
sns.factorplot('Pclass','Survived',hue='Sex',data=data)
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_18_0.png">


crosstab과 factorplot을 살펴보면 Pclass=1에서 female의 사망자수가 3명이고,   
생존율은 약 95-96%임을 볼 수 있습니다.

## Age


```python
data['Age'].describe()
```




    count    714.000000
    mean      29.699118
    std       14.526497
    min        0.420000
    25%       20.125000
    50%       28.000000
    75%       38.000000
    max       80.000000
    Name: Age, dtype: float64



우리는 위에서 Age에 177개의 결측값이 있음을 확인 했습니다.  
어떻게 해결할 수 있을까요?  
단순히 평균,중앙값으로 채울 수도 있겠지만
Name column에 Mr,Mrs를 이용하여 Age를 채워 봅시다.

정규표현식을 이용하여 키워드를 추출합시다.  
정규표현식을 잠깐 설명하자면
[A-za-z] 는 각각 A-Z,a-z대소문자중의 하나를 뜻하고  
+기호는 그 앞에 있는것이 최소한 한번은 나와야 함을 뜻합니다.  
\\.기호는 마침표를 뜻합니다.  


```python
data['Initial']=0
for i in data:
    data['Initial'] = data.Name.str.extract('([A-za-z]+)\.')
```


```python
pd.crosstab(data.Initial,data.Sex).T.style.background_gradient(cmap='summer_r')
```




<style  type="text/css" >
    #T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col0 {
            background-color:  #ffff66;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col1 {
            background-color:  #ffff66;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col2 {
            background-color:  #008066;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col3 {
            background-color:  #ffff66;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col4 {
            background-color:  #ffff66;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col5 {
            background-color:  #ffff66;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col6 {
            background-color:  #008066;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col7 {
            background-color:  #ffff66;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col8 {
            background-color:  #ffff66;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col9 {
            background-color:  #008066;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col10 {
            background-color:  #008066;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col11 {
            background-color:  #008066;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col12 {
            background-color:  #ffff66;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col13 {
            background-color:  #008066;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col14 {
            background-color:  #008066;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col15 {
            background-color:  #ffff66;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col16 {
            background-color:  #ffff66;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col0 {
            background-color:  #008066;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col1 {
            background-color:  #008066;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col2 {
            background-color:  #ffff66;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col3 {
            background-color:  #008066;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col4 {
            background-color:  #008066;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col5 {
            background-color:  #008066;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col6 {
            background-color:  #ffff66;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col7 {
            background-color:  #008066;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col8 {
            background-color:  #008066;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col9 {
            background-color:  #ffff66;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col10 {
            background-color:  #ffff66;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col11 {
            background-color:  #ffff66;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col12 {
            background-color:  #008066;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col13 {
            background-color:  #ffff66;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col14 {
            background-color:  #ffff66;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col15 {
            background-color:  #008066;
        }    #T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col16 {
            background-color:  #008066;
        }</style>  
<table id="T_c675feac_f0ac_11e9_98a0_bc838524d564" > 
<thead>    <tr> 
        <th class="index_name level0" >Initial</th> 
        <th class="col_heading level0 col0" >Capt</th> 
        <th class="col_heading level0 col1" >Col</th> 
        <th class="col_heading level0 col2" >Countess</th> 
        <th class="col_heading level0 col3" >Don</th> 
        <th class="col_heading level0 col4" >Dr</th> 
        <th class="col_heading level0 col5" >Jonkheer</th> 
        <th class="col_heading level0 col6" >Lady</th> 
        <th class="col_heading level0 col7" >Major</th> 
        <th class="col_heading level0 col8" >Master</th> 
        <th class="col_heading level0 col9" >Miss</th> 
        <th class="col_heading level0 col10" >Mlle</th> 
        <th class="col_heading level0 col11" >Mme</th> 
        <th class="col_heading level0 col12" >Mr</th> 
        <th class="col_heading level0 col13" >Mrs</th> 
        <th class="col_heading level0 col14" >Ms</th> 
        <th class="col_heading level0 col15" >Rev</th> 
        <th class="col_heading level0 col16" >Sir</th> 
    </tr>    <tr> 
        <th class="index_name level0" >Sex</th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_c675feac_f0ac_11e9_98a0_bc838524d564level0_row0" class="row_heading level0 row0" >female</th> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col0" class="data row0 col0" >0</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col1" class="data row0 col1" >0</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col2" class="data row0 col2" >1</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col3" class="data row0 col3" >0</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col4" class="data row0 col4" >1</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col5" class="data row0 col5" >0</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col6" class="data row0 col6" >1</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col7" class="data row0 col7" >0</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col8" class="data row0 col8" >0</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col9" class="data row0 col9" >182</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col10" class="data row0 col10" >2</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col11" class="data row0 col11" >1</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col12" class="data row0 col12" >0</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col13" class="data row0 col13" >125</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col14" class="data row0 col14" >1</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col15" class="data row0 col15" >0</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row0_col16" class="data row0 col16" >0</td> 
    </tr>    <tr> 
        <th id="T_c675feac_f0ac_11e9_98a0_bc838524d564level0_row1" class="row_heading level0 row1" >male</th> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col0" class="data row1 col0" >1</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col1" class="data row1 col1" >2</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col2" class="data row1 col2" >0</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col3" class="data row1 col3" >1</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col4" class="data row1 col4" >6</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col5" class="data row1 col5" >1</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col6" class="data row1 col6" >0</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col7" class="data row1 col7" >2</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col8" class="data row1 col8" >40</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col9" class="data row1 col9" >0</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col10" class="data row1 col10" >0</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col11" class="data row1 col11" >0</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col12" class="data row1 col12" >517</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col13" class="data row1 col13" >0</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col14" class="data row1 col14" >0</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col15" class="data row1 col15" >6</td> 
        <td id="T_c675feac_f0ac_11e9_98a0_bc838524d564row1_col16" class="data row1 col16" >1</td> 
    </tr></tbody> 
</table> 



Mlle Mme등 오타가 있으므로 수정해봅시다.


```python
data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
```


```python
data.groupby('Initial')['Age'].mean()
```




    Initial
    Master     4.574167
    Miss      21.860000
    Mr        32.739609
    Mrs       35.981818
    Other     45.888889
    Name: Age, dtype: float64



위에서 구한 Initial별 Age의 평균으로 결측값을 채워 봅시다.


```python
data.loc[(data.Age.isnull())&(data.Initial=='Mr'),'Age']=33
data.loc[(data.Age.isnull())&(data.Initial=='Mrs'),'Age']=36
data.loc[(data.Age.isnull())&(data.Initial=='Master'),'Age']=5
data.loc[(data.Age.isnull())&(data.Initial=='Miss'),'Age']=22
data.loc[(data.Age.isnull())&(data.Initial=='Other'),'Age']=46
```


```python
data.Age.isnull().any()
```




    False




```python
f, ax = plt.subplots(figsize=(18,8))
sns.distplot(data['Age'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1761f187128>



<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_32_1.png">



```python
f,ax=plt.subplots(1,2,figsize=(20,10))
data[data['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
data[data['Survived']==1].Age.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')
ax[1].set_title('Survived= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_33_0.png">



```python
f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass","Age", hue="Survived", data=data,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Sex","Age", hue="Survived", data=data,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_34_0.png">



```python
sns.factorplot('Pclass','Survived',col='Initial',data=data)
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_35_0.png">


## Embarked


```python
pd.crosstab([data.Embarked,data.Pclass],[data.Sex,data.Survived],margins=True).style.background_gradient(cmap='summer_r')
```




<style  type="text/css" >
    #T_fba70300_f0ac_11e9_806c_bc838524d564row0_col0 {
            background-color:  #fcfe66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row0_col1 {
            background-color:  #d2e866;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row0_col2 {
            background-color:  #f2f866;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row0_col3 {
            background-color:  #d8ec66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row0_col4 {
            background-color:  #e8f466;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row1_col0 {
            background-color:  #ffff66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row1_col1 {
            background-color:  #f9fc66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row1_col2 {
            background-color:  #fcfe66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row1_col3 {
            background-color:  #fbfd66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row1_col4 {
            background-color:  #fbfd66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row2_col0 {
            background-color:  #e6f266;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row2_col1 {
            background-color:  #f0f866;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row2_col2 {
            background-color:  #eef666;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row2_col3 {
            background-color:  #e8f466;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row2_col4 {
            background-color:  #edf666;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row3_col0 {
            background-color:  #ffff66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row3_col1 {
            background-color:  #ffff66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row3_col2 {
            background-color:  #ffff66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row3_col3 {
            background-color:  #ffff66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row3_col4 {
            background-color:  #ffff66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row4_col0 {
            background-color:  #ffff66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row4_col1 {
            background-color:  #fefe66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row4_col2 {
            background-color:  #ffff66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row4_col3 {
            background-color:  #ffff66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row4_col4 {
            background-color:  #ffff66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row5_col0 {
            background-color:  #e3f166;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row5_col1 {
            background-color:  #e6f266;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row5_col2 {
            background-color:  #ecf666;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row5_col3 {
            background-color:  #f8fc66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row5_col4 {
            background-color:  #ebf566;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row6_col0 {
            background-color:  #f9fc66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row6_col1 {
            background-color:  #cde666;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row6_col2 {
            background-color:  #e4f266;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row6_col3 {
            background-color:  #bede66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row6_col4 {
            background-color:  #dbed66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row7_col0 {
            background-color:  #edf666;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row7_col1 {
            background-color:  #bdde66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row7_col2 {
            background-color:  #d3e966;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row7_col3 {
            background-color:  #dcee66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row7_col4 {
            background-color:  #d1e866;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row8_col0 {
            background-color:  #52a866;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row8_col1 {
            background-color:  #dcee66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row8_col2 {
            background-color:  #81c066;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row8_col3 {
            background-color:  #b0d866;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row8_col4 {
            background-color:  #9acc66;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row9_col0 {
            background-color:  #008066;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row9_col1 {
            background-color:  #008066;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row9_col2 {
            background-color:  #008066;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row9_col3 {
            background-color:  #008066;
        }    #T_fba70300_f0ac_11e9_806c_bc838524d564row9_col4 {
            background-color:  #008066;
        }</style>  
<table id="T_fba70300_f0ac_11e9_806c_bc838524d564" > 
<thead>    <tr> 
        <th class="blank" ></th> 
        <th class="index_name level0" >Sex</th> 
        <th class="col_heading level0 col0" colspan=2>female</th> 
        <th class="col_heading level0 col2" colspan=2>male</th> 
        <th class="col_heading level0 col4" >All</th> 
    </tr>    <tr> 
        <th class="blank" ></th> 
        <th class="index_name level1" >Survived</th> 
        <th class="col_heading level1 col0" >0</th> 
        <th class="col_heading level1 col1" >1</th> 
        <th class="col_heading level1 col2" >0</th> 
        <th class="col_heading level1 col3" >1</th> 
        <th class="col_heading level1 col4" ></th> 
    </tr>    <tr> 
        <th class="index_name level0" >Embarked</th> 
        <th class="index_name level1" >Pclass</th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_fba70300_f0ac_11e9_806c_bc838524d564level0_row0" class="row_heading level0 row0" >C</th> 
        <th id="T_fba70300_f0ac_11e9_806c_bc838524d564level1_row0" class="row_heading level1 row0" >1</th> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row0_col0" class="data row0 col0" >1</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row0_col1" class="data row0 col1" >42</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row0_col2" class="data row0 col2" >25</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row0_col3" class="data row0 col3" >17</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row0_col4" class="data row0 col4" >85</td> 
    </tr>    <tr> 
        <th id="T_fba70300_f0ac_11e9_806c_bc838524d564level0_row0" class="row_heading level0 row0" >C</th>
        <th id="T_fba70300_f0ac_11e9_806c_bc838524d564level1_row1" class="row_heading level1 row1" >2</th> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row1_col0" class="data row1 col0" >0</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row1_col1" class="data row1 col1" >7</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row1_col2" class="data row1 col2" >8</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row1_col3" class="data row1 col3" >2</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row1_col4" class="data row1 col4" >17</td> 
    </tr>    <tr> 
        <th id="T_fba70300_f0ac_11e9_806c_bc838524d564level0_row0" class="row_heading level0 row0" >C</th>
        <th id="T_fba70300_f0ac_11e9_806c_bc838524d564level1_row2" class="row_heading level1 row2" >3</th> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row2_col0" class="data row2 col0" >8</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row2_col1" class="data row2 col1" >15</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row2_col2" class="data row2 col2" >33</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row2_col3" class="data row2 col3" >10</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row2_col4" class="data row2 col4" >66</td> 
    </tr>    <tr> 
        <th id="T_fba70300_f0ac_11e9_806c_bc838524d564level0_row3" class="row_heading level0 row3" >Q</th> 
        <th id="T_fba70300_f0ac_11e9_806c_bc838524d564level1_row3" class="row_heading level1 row3" >1</th> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row3_col0" class="data row3 col0" >0</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row3_col1" class="data row3 col1" >1</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row3_col2" class="data row3 col2" >1</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row3_col3" class="data row3 col3" >0</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row3_col4" class="data row3 col4" >2</td> 
    </tr>    <tr> 
        <th id="T_fba70300_f0ac_11e9_806c_bc838524d564level0_row3" class="row_heading level0 row3" >Q</th>
        <th id="T_fba70300_f0ac_11e9_806c_bc838524d564level1_row4" class="row_heading level1 row4" >2</th> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row4_col0" class="data row4 col0" >0</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row4_col1" class="data row4 col1" >2</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row4_col2" class="data row4 col2" >1</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row4_col3" class="data row4 col3" >0</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row4_col4" class="data row4 col4" >3</td> 
    </tr>    <tr> 
        <th id="T_fba70300_f0ac_11e9_806c_bc838524d564level0_row3" class="row_heading level0 row3" >Q</th>
        <th id="T_fba70300_f0ac_11e9_806c_bc838524d564level1_row5" class="row_heading level1 row5" >3</th> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row5_col0" class="data row5 col0" >9</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row5_col1" class="data row5 col1" >24</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row5_col2" class="data row5 col2" >36</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row5_col3" class="data row5 col3" >3</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row5_col4" class="data row5 col4" >72</td> 
    </tr>    <tr> 
        <th id="T_fba70300_f0ac_11e9_806c_bc838524d564level0_row6" class="row_heading level0 row6" >S</th> 
        <th id="T_fba70300_f0ac_11e9_806c_bc838524d564level1_row6" class="row_heading level1 row6" >1</th> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row6_col0" class="data row6 col0" >2</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row6_col1" class="data row6 col1" >46</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row6_col2" class="data row6 col2" >51</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row6_col3" class="data row6 col3" >28</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row6_col4" class="data row6 col4" >127</td> 
    </tr>    <tr> 
        <th id="T_fba70300_f0ac_11e9_806c_bc838524d564level0_row6" class="row_heading level0 row6" >S</th> 
        <th id="T_fba70300_f0ac_11e9_806c_bc838524d564level1_row7" class="row_heading level1 row7" >2</th> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row7_col0" class="data row7 col0" >6</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row7_col1" class="data row7 col1" >61</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row7_col2" class="data row7 col2" >82</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row7_col3" class="data row7 col3" >15</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row7_col4" class="data row7 col4" >164</td> 
    </tr>    <tr> 
        <th id="T_fba70300_f0ac_11e9_806c_bc838524d564level0_row6" class="row_heading level0 row6" >S</th> 
        <th id="T_fba70300_f0ac_11e9_806c_bc838524d564level1_row8" class="row_heading level1 row8" >3</th> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row8_col0" class="data row8 col0" >55</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row8_col1" class="data row8 col1" >33</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row8_col2" class="data row8 col2" >231</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row8_col3" class="data row8 col3" >34</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row8_col4" class="data row8 col4" >353</td> 
    </tr>    <tr> 
        <th id="T_fba70300_f0ac_11e9_806c_bc838524d564level0_row9" class="row_heading level0 row9" >All</th> 
        <th id="T_fba70300_f0ac_11e9_806c_bc838524d564level1_row9" class="row_heading level1 row9" ></th> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row9_col0" class="data row9 col0" >81</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row9_col1" class="data row9 col1" >231</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row9_col2" class="data row9 col2" >468</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row9_col3" class="data row9 col3" >109</td> 
        <td id="T_fba70300_f0ac_11e9_806c_bc838524d564row9_col4" class="data row9 col4" >889</td> 
    </tr></tbody> 
</table> 




```python
sns.factorplot('Embarked','Survived',data=data)
fig=plt.gcf()
fig.set_size_inches(5,3)
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_38_0.png">



```python
f,ax = plt.subplots(2,2,figsize=(20,15))
sns.countplot('Embarked',data=data,ax=ax[0,0])
ax[0,0].set_title('No. Of Passengers Boarded')
ax[0,0].set_ylabel('')
sns.countplot('Embarked',hue='Sex',data=data,ax=ax[0,1])
ax[0,1].set_title('Male-Female Split for Embarked')
ax[0,1].set_ylabel('')
sns.countplot('Embarked',hue='Survived',data=data,ax=ax[1,0])
ax[1,0].set_title('Embarked vs Survivied')
ax[1,0].set_ylabel('')
sns.countplot('Embarked',hue='Pclass',data=data,ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')
ax[1,1].set_ylabel('')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_39_0.png">



```python
sns.factorplot('Pclass','Survived',hue='Sex',col='Embarked',data=data)
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_40_0.png">


Embarked에도 결측값이 2개가 있었는데 가장 승객이많았던 S로 결측값을 채웁니다.


```python
data['Embarked'].fillna('S',inplace=True)
```


```python
data.Embarked.isnull().any()
```




    False



## SibSip


```python
pd.crosstab([data.SibSp],data.Survived).style.background_gradient(cmap='summer_r')
```




<style  type="text/css" >
    #T_1bdecc0c_f0ad_11e9_8689_bc838524d564row0_col0 {
            background-color:  #008066;
        }    #T_1bdecc0c_f0ad_11e9_8689_bc838524d564row0_col1 {
            background-color:  #008066;
        }    #T_1bdecc0c_f0ad_11e9_8689_bc838524d564row1_col0 {
            background-color:  #c4e266;
        }    #T_1bdecc0c_f0ad_11e9_8689_bc838524d564row1_col1 {
            background-color:  #77bb66;
        }    #T_1bdecc0c_f0ad_11e9_8689_bc838524d564row2_col0 {
            background-color:  #f9fc66;
        }    #T_1bdecc0c_f0ad_11e9_8689_bc838524d564row2_col1 {
            background-color:  #f0f866;
        }    #T_1bdecc0c_f0ad_11e9_8689_bc838524d564row3_col0 {
            background-color:  #fbfd66;
        }    #T_1bdecc0c_f0ad_11e9_8689_bc838524d564row3_col1 {
            background-color:  #fbfd66;
        }    #T_1bdecc0c_f0ad_11e9_8689_bc838524d564row4_col0 {
            background-color:  #f9fc66;
        }    #T_1bdecc0c_f0ad_11e9_8689_bc838524d564row4_col1 {
            background-color:  #fcfe66;
        }    #T_1bdecc0c_f0ad_11e9_8689_bc838524d564row5_col0 {
            background-color:  #ffff66;
        }    #T_1bdecc0c_f0ad_11e9_8689_bc838524d564row5_col1 {
            background-color:  #ffff66;
        }    #T_1bdecc0c_f0ad_11e9_8689_bc838524d564row6_col0 {
            background-color:  #fefe66;
        }    #T_1bdecc0c_f0ad_11e9_8689_bc838524d564row6_col1 {
            background-color:  #ffff66;
        }</style>  
<table id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564" > 
<thead>    <tr> 
        <th class="index_name level0" >Survived</th> 
        <th class="col_heading level0 col0" >0</th> 
        <th class="col_heading level0 col1" >1</th> 
    </tr>    <tr> 
        <th class="index_name level0" >SibSp</th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564level0_row0" class="row_heading level0 row0" >0</th> 
        <td id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564row0_col0" class="data row0 col0" >398</td> 
        <td id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564row0_col1" class="data row0 col1" >210</td> 
    </tr>    <tr> 
        <th id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564level0_row1" class="row_heading level0 row1" >1</th> 
        <td id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564row1_col0" class="data row1 col0" >97</td> 
        <td id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564row1_col1" class="data row1 col1" >112</td> 
    </tr>    <tr> 
        <th id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564level0_row2" class="row_heading level0 row2" >2</th> 
        <td id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564row2_col0" class="data row2 col0" >15</td> 
        <td id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564row2_col1" class="data row2 col1" >13</td> 
    </tr>    <tr> 
        <th id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564level0_row3" class="row_heading level0 row3" >3</th> 
        <td id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564row3_col0" class="data row3 col0" >12</td> 
        <td id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564row3_col1" class="data row3 col1" >4</td> 
    </tr>    <tr> 
        <th id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564level0_row4" class="row_heading level0 row4" >4</th> 
        <td id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564row4_col0" class="data row4 col0" >15</td> 
        <td id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564row4_col1" class="data row4 col1" >3</td> 
    </tr>    <tr> 
        <th id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564level0_row5" class="row_heading level0 row5" >5</th> 
        <td id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564row5_col0" class="data row5 col0" >5</td> 
        <td id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564row5_col1" class="data row5 col1" >0</td> 
    </tr>    <tr> 
        <th id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564level0_row6" class="row_heading level0 row6" >8</th> 
        <td id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564row6_col0" class="data row6 col0" >7</td> 
        <td id="T_1bdecc0c_f0ad_11e9_8689_bc838524d564row6_col1" class="data row6 col1" >0</td> 
    </tr></tbody> 
</table> 




```python
f, ax=plt.subplots(1,2,figsize=(20,10))
sns.barplot('SibSp','Survived',data=data,ax=ax[0])
ax[0].set_title('SibSp vs Survived')
sns.factorplot('SibSp','Survived',data=data,ax=ax[1])
ax[1].set_title('SibSp vs Survived')
plt.close(2)
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_46_0.png">


그래프를보면 혼자탑승하거나 두명이서 탑승한 경우가 생존율이 가장 높습니다.


```python
pd.crosstab(data.SibSp,data.Pclass).style.background_gradient(cmap='summer_r')
```




<style  type="text/css" >
    #T_277bdfac_f0ad_11e9_931c_bc838524d564row0_col0 {
            background-color:  #008066;
        }    #T_277bdfac_f0ad_11e9_931c_bc838524d564row0_col1 {
            background-color:  #008066;
        }    #T_277bdfac_f0ad_11e9_931c_bc838524d564row0_col2 {
            background-color:  #008066;
        }    #T_277bdfac_f0ad_11e9_931c_bc838524d564row1_col0 {
            background-color:  #7bbd66;
        }    #T_277bdfac_f0ad_11e9_931c_bc838524d564row1_col1 {
            background-color:  #8ac466;
        }    #T_277bdfac_f0ad_11e9_931c_bc838524d564row1_col2 {
            background-color:  #c6e266;
        }    #T_277bdfac_f0ad_11e9_931c_bc838524d564row2_col0 {
            background-color:  #f6fa66;
        }    #T_277bdfac_f0ad_11e9_931c_bc838524d564row2_col1 {
            background-color:  #eef666;
        }    #T_277bdfac_f0ad_11e9_931c_bc838524d564row2_col2 {
            background-color:  #f8fc66;
        }    #T_277bdfac_f0ad_11e9_931c_bc838524d564row3_col0 {
            background-color:  #fafc66;
        }    #T_277bdfac_f0ad_11e9_931c_bc838524d564row3_col1 {
            background-color:  #fdfe66;
        }    #T_277bdfac_f0ad_11e9_931c_bc838524d564row3_col2 {
            background-color:  #fafc66;
        }    #T_277bdfac_f0ad_11e9_931c_bc838524d564row4_col0 {
            background-color:  #ffff66;
        }    #T_277bdfac_f0ad_11e9_931c_bc838524d564row4_col1 {
            background-color:  #ffff66;
        }    #T_277bdfac_f0ad_11e9_931c_bc838524d564row4_col2 {
            background-color:  #f6fa66;
        }    #T_277bdfac_f0ad_11e9_931c_bc838524d564row5_col0 {
            background-color:  #ffff66;
        }    #T_277bdfac_f0ad_11e9_931c_bc838524d564row5_col1 {
            background-color:  #ffff66;
        }    #T_277bdfac_f0ad_11e9_931c_bc838524d564row5_col2 {
            background-color:  #ffff66;
        }    #T_277bdfac_f0ad_11e9_931c_bc838524d564row6_col0 {
            background-color:  #ffff66;
        }    #T_277bdfac_f0ad_11e9_931c_bc838524d564row6_col1 {
            background-color:  #ffff66;
        }    #T_277bdfac_f0ad_11e9_931c_bc838524d564row6_col2 {
            background-color:  #fefe66;
        }</style>  
<table id="T_277bdfac_f0ad_11e9_931c_bc838524d564" > 
<thead>    <tr> 
        <th class="index_name level0" >Pclass</th> 
        <th class="col_heading level0 col0" >1</th> 
        <th class="col_heading level0 col1" >2</th> 
        <th class="col_heading level0 col2" >3</th> 
    </tr>    <tr> 
        <th class="index_name level0" >SibSp</th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_277bdfac_f0ad_11e9_931c_bc838524d564level0_row0" class="row_heading level0 row0" >0</th> 
        <td id="T_277bdfac_f0ad_11e9_931c_bc838524d564row0_col0" class="data row0 col0" >137</td> 
        <td id="T_277bdfac_f0ad_11e9_931c_bc838524d564row0_col1" class="data row0 col1" >120</td> 
        <td id="T_277bdfac_f0ad_11e9_931c_bc838524d564row0_col2" class="data row0 col2" >351</td> 
    </tr>    <tr> 
        <th id="T_277bdfac_f0ad_11e9_931c_bc838524d564level0_row1" class="row_heading level0 row1" >1</th> 
        <td id="T_277bdfac_f0ad_11e9_931c_bc838524d564row1_col0" class="data row1 col0" >71</td> 
        <td id="T_277bdfac_f0ad_11e9_931c_bc838524d564row1_col1" class="data row1 col1" >55</td> 
        <td id="T_277bdfac_f0ad_11e9_931c_bc838524d564row1_col2" class="data row1 col2" >83</td> 
    </tr>    <tr> 
        <th id="T_277bdfac_f0ad_11e9_931c_bc838524d564level0_row2" class="row_heading level0 row2" >2</th> 
        <td id="T_277bdfac_f0ad_11e9_931c_bc838524d564row2_col0" class="data row2 col0" >5</td> 
        <td id="T_277bdfac_f0ad_11e9_931c_bc838524d564row2_col1" class="data row2 col1" >8</td> 
        <td id="T_277bdfac_f0ad_11e9_931c_bc838524d564row2_col2" class="data row2 col2" >15</td> 
    </tr>    <tr> 
        <th id="T_277bdfac_f0ad_11e9_931c_bc838524d564level0_row3" class="row_heading level0 row3" >3</th> 
        <td id="T_277bdfac_f0ad_11e9_931c_bc838524d564row3_col0" class="data row3 col0" >3</td> 
        <td id="T_277bdfac_f0ad_11e9_931c_bc838524d564row3_col1" class="data row3 col1" >1</td> 
        <td id="T_277bdfac_f0ad_11e9_931c_bc838524d564row3_col2" class="data row3 col2" >12</td> 
    </tr>    <tr> 
        <th id="T_277bdfac_f0ad_11e9_931c_bc838524d564level0_row4" class="row_heading level0 row4" >4</th> 
        <td id="T_277bdfac_f0ad_11e9_931c_bc838524d564row4_col0" class="data row4 col0" >0</td> 
        <td id="T_277bdfac_f0ad_11e9_931c_bc838524d564row4_col1" class="data row4 col1" >0</td> 
        <td id="T_277bdfac_f0ad_11e9_931c_bc838524d564row4_col2" class="data row4 col2" >18</td> 
    </tr>    <tr> 
        <th id="T_277bdfac_f0ad_11e9_931c_bc838524d564level0_row5" class="row_heading level0 row5" >5</th> 
        <td id="T_277bdfac_f0ad_11e9_931c_bc838524d564row5_col0" class="data row5 col0" >0</td> 
        <td id="T_277bdfac_f0ad_11e9_931c_bc838524d564row5_col1" class="data row5 col1" >0</td> 
        <td id="T_277bdfac_f0ad_11e9_931c_bc838524d564row5_col2" class="data row5 col2" >5</td> 
    </tr>    <tr> 
        <th id="T_277bdfac_f0ad_11e9_931c_bc838524d564level0_row6" class="row_heading level0 row6" >8</th> 
        <td id="T_277bdfac_f0ad_11e9_931c_bc838524d564row6_col0" class="data row6 col0" >0</td> 
        <td id="T_277bdfac_f0ad_11e9_931c_bc838524d564row6_col1" class="data row6 col1" >0</td> 
        <td id="T_277bdfac_f0ad_11e9_931c_bc838524d564row6_col2" class="data row6 col2" >7</td> 
    </tr></tbody> 
</table> 



3인이상 승객의경우 Pclass가 3이므로 죽을 확률이 높습니다.

## Parch


```python
pd.crosstab(data.Parch,data.Pclass).style.background_gradient(cmap='summer_r')
```




<style  type="text/css" >
    #T_35779552_f0ad_11e9_9ca5_bc838524d564row0_col0 {
            background-color:  #008066;
        }    #T_35779552_f0ad_11e9_9ca5_bc838524d564row0_col1 {
            background-color:  #008066;
        }    #T_35779552_f0ad_11e9_9ca5_bc838524d564row0_col2 {
            background-color:  #008066;
        }    #T_35779552_f0ad_11e9_9ca5_bc838524d564row1_col0 {
            background-color:  #cfe766;
        }    #T_35779552_f0ad_11e9_9ca5_bc838524d564row1_col1 {
            background-color:  #c2e066;
        }    #T_35779552_f0ad_11e9_9ca5_bc838524d564row1_col2 {
            background-color:  #dbed66;
        }    #T_35779552_f0ad_11e9_9ca5_bc838524d564row2_col0 {
            background-color:  #dfef66;
        }    #T_35779552_f0ad_11e9_9ca5_bc838524d564row2_col1 {
            background-color:  #e1f066;
        }    #T_35779552_f0ad_11e9_9ca5_bc838524d564row2_col2 {
            background-color:  #e3f166;
        }    #T_35779552_f0ad_11e9_9ca5_bc838524d564row3_col0 {
            background-color:  #ffff66;
        }    #T_35779552_f0ad_11e9_9ca5_bc838524d564row3_col1 {
            background-color:  #fcfe66;
        }    #T_35779552_f0ad_11e9_9ca5_bc838524d564row3_col2 {
            background-color:  #fefe66;
        }    #T_35779552_f0ad_11e9_9ca5_bc838524d564row4_col0 {
            background-color:  #fefe66;
        }    #T_35779552_f0ad_11e9_9ca5_bc838524d564row4_col1 {
            background-color:  #ffff66;
        }    #T_35779552_f0ad_11e9_9ca5_bc838524d564row4_col2 {
            background-color:  #fefe66;
        }    #T_35779552_f0ad_11e9_9ca5_bc838524d564row5_col0 {
            background-color:  #ffff66;
        }    #T_35779552_f0ad_11e9_9ca5_bc838524d564row5_col1 {
            background-color:  #ffff66;
        }    #T_35779552_f0ad_11e9_9ca5_bc838524d564row5_col2 {
            background-color:  #fdfe66;
        }    #T_35779552_f0ad_11e9_9ca5_bc838524d564row6_col0 {
            background-color:  #ffff66;
        }    #T_35779552_f0ad_11e9_9ca5_bc838524d564row6_col1 {
            background-color:  #ffff66;
        }    #T_35779552_f0ad_11e9_9ca5_bc838524d564row6_col2 {
            background-color:  #ffff66;
        }</style>  
<table id="T_35779552_f0ad_11e9_9ca5_bc838524d564" > 
<thead>    <tr> 
        <th class="index_name level0" >Pclass</th> 
        <th class="col_heading level0 col0" >1</th> 
        <th class="col_heading level0 col1" >2</th> 
        <th class="col_heading level0 col2" >3</th> 
    </tr>    <tr> 
        <th class="index_name level0" >Parch</th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_35779552_f0ad_11e9_9ca5_bc838524d564level0_row0" class="row_heading level0 row0" >0</th> 
        <td id="T_35779552_f0ad_11e9_9ca5_bc838524d564row0_col0" class="data row0 col0" >163</td> 
        <td id="T_35779552_f0ad_11e9_9ca5_bc838524d564row0_col1" class="data row0 col1" >134</td> 
        <td id="T_35779552_f0ad_11e9_9ca5_bc838524d564row0_col2" class="data row0 col2" >381</td> 
    </tr>    <tr> 
        <th id="T_35779552_f0ad_11e9_9ca5_bc838524d564level0_row1" class="row_heading level0 row1" >1</th> 
        <td id="T_35779552_f0ad_11e9_9ca5_bc838524d564row1_col0" class="data row1 col0" >31</td> 
        <td id="T_35779552_f0ad_11e9_9ca5_bc838524d564row1_col1" class="data row1 col1" >32</td> 
        <td id="T_35779552_f0ad_11e9_9ca5_bc838524d564row1_col2" class="data row1 col2" >55</td> 
    </tr>    <tr> 
        <th id="T_35779552_f0ad_11e9_9ca5_bc838524d564level0_row2" class="row_heading level0 row2" >2</th> 
        <td id="T_35779552_f0ad_11e9_9ca5_bc838524d564row2_col0" class="data row2 col0" >21</td> 
        <td id="T_35779552_f0ad_11e9_9ca5_bc838524d564row2_col1" class="data row2 col1" >16</td> 
        <td id="T_35779552_f0ad_11e9_9ca5_bc838524d564row2_col2" class="data row2 col2" >43</td> 
    </tr>    <tr> 
        <th id="T_35779552_f0ad_11e9_9ca5_bc838524d564level0_row3" class="row_heading level0 row3" >3</th> 
        <td id="T_35779552_f0ad_11e9_9ca5_bc838524d564row3_col0" class="data row3 col0" >0</td> 
        <td id="T_35779552_f0ad_11e9_9ca5_bc838524d564row3_col1" class="data row3 col1" >2</td> 
        <td id="T_35779552_f0ad_11e9_9ca5_bc838524d564row3_col2" class="data row3 col2" >3</td> 
    </tr>    <tr> 
        <th id="T_35779552_f0ad_11e9_9ca5_bc838524d564level0_row4" class="row_heading level0 row4" >4</th> 
        <td id="T_35779552_f0ad_11e9_9ca5_bc838524d564row4_col0" class="data row4 col0" >1</td> 
        <td id="T_35779552_f0ad_11e9_9ca5_bc838524d564row4_col1" class="data row4 col1" >0</td> 
        <td id="T_35779552_f0ad_11e9_9ca5_bc838524d564row4_col2" class="data row4 col2" >3</td> 
    </tr>    <tr> 
        <th id="T_35779552_f0ad_11e9_9ca5_bc838524d564level0_row5" class="row_heading level0 row5" >5</th> 
        <td id="T_35779552_f0ad_11e9_9ca5_bc838524d564row5_col0" class="data row5 col0" >0</td> 
        <td id="T_35779552_f0ad_11e9_9ca5_bc838524d564row5_col1" class="data row5 col1" >0</td> 
        <td id="T_35779552_f0ad_11e9_9ca5_bc838524d564row5_col2" class="data row5 col2" >5</td> 
    </tr>    <tr> 
        <th id="T_35779552_f0ad_11e9_9ca5_bc838524d564level0_row6" class="row_heading level0 row6" >6</th> 
        <td id="T_35779552_f0ad_11e9_9ca5_bc838524d564row6_col0" class="data row6 col0" >0</td> 
        <td id="T_35779552_f0ad_11e9_9ca5_bc838524d564row6_col1" class="data row6 col1" >0</td> 
        <td id="T_35779552_f0ad_11e9_9ca5_bc838524d564row6_col2" class="data row6 col2" >1</td> 
    </tr></tbody> 
</table> 



Parch feature 또한 많은 가족인 경우 Pclass=3 임을 볼 수 있습니다.


```python
f,ax=plt.subplots(1,2,figsize=(20,8))
sns.barplot('Parch','Survived',data=data,ax=ax[0])
ax[0].set_title('Parch vs Survived')
sns.factorplot('Parch','Survived',data=data,ax=ax[1])
ax[1].set_title('Parch vs Survived')
plt.close(2)
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_53_0.png">


부모와 함께 탑승한 승객은 생존 가능성이 더 높습니다.  
하지만 Parch >4 이상이면 생존 가능성이 줄어듭니다.

## Fare


```python
data['Fare'].describe()
```




    count    891.000000
    mean      32.204208
    std       49.693429
    min        0.000000
    25%        7.910400
    50%       14.454200
    75%       31.000000
    max      512.329200
    Name: Fare, dtype: float64



무임 승차 고객이 있는게 보입니다.

Pclass 별 Fare 분포표


```python
f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(data[data['Pclass']==1].Fare,ax=ax[0])
ax[0].set_title('Fares in Pclass 1')
sns.distplot(data[data['Pclass']==2].Fare,ax=ax[1])
ax[1].set_title('Fares in Pclass 2')
sns.distplot(data[data['Pclass']==3].Fare,ax=ax[2])
ax[2].set_title('Fares in Pclass 3')
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_59_0.png">


## 특징 요약
Sex: 여성이면 일단 생존 확률이 높다.  
Pclass: 1,2등석이면 생존 확률이 높다.  
Age: 영유아의 경우 생존 확률이 높다.  
Embarked: 대다수의 Pclass=1의 승객은 S이지만 C에서의 생존확률이 더 높다. Q의 승객은 모두 Pclass=3.  
Parch+SibSp: 1~2명의 형제, 배우자 또는 1~3명의 가족과있는것이 혼자있거나 대가족과 동행하는 것 보다 생존 확률이 높다.


```python
sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_61_0.png">


# Feature Engineering

## Age_band

연령을 16살 단위로 쪼개서 5개의 class로 분리합니다.


```python
data['Age_band']=0
data.loc[data['Age']<=16,'Age_band']=0
data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1
data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2
data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3
data.loc[data['Age']>64,'Age_band']=4
data.head(2)
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Initial</th>
      <th>Age_band</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>Mrs</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
data['Age_band'].value_counts().to_frame().style.background_gradient(cmap='summer')
```




<style  type="text/css" >
    #T_74a7695c_f0ad_11e9_ab3d_bc838524d564row0_col0 {
            background-color:  #ffff66;
        }    #T_74a7695c_f0ad_11e9_ab3d_bc838524d564row1_col0 {
            background-color:  #d8ec66;
        }    #T_74a7695c_f0ad_11e9_ab3d_bc838524d564row2_col0 {
            background-color:  #40a066;
        }    #T_74a7695c_f0ad_11e9_ab3d_bc838524d564row3_col0 {
            background-color:  #289366;
        }    #T_74a7695c_f0ad_11e9_ab3d_bc838524d564row4_col0 {
            background-color:  #008066;
        }</style>  
<table id="T_74a7695c_f0ad_11e9_ab3d_bc838524d564" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >Age_band</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_74a7695c_f0ad_11e9_ab3d_bc838524d564level0_row0" class="row_heading level0 row0" >1</th> 
        <td id="T_74a7695c_f0ad_11e9_ab3d_bc838524d564row0_col0" class="data row0 col0" >382</td> 
    </tr>    <tr> 
        <th id="T_74a7695c_f0ad_11e9_ab3d_bc838524d564level0_row1" class="row_heading level0 row1" >2</th> 
        <td id="T_74a7695c_f0ad_11e9_ab3d_bc838524d564row1_col0" class="data row1 col0" >325</td> 
    </tr>    <tr> 
        <th id="T_74a7695c_f0ad_11e9_ab3d_bc838524d564level0_row2" class="row_heading level0 row2" >0</th> 
        <td id="T_74a7695c_f0ad_11e9_ab3d_bc838524d564row2_col0" class="data row2 col0" >104</td> 
    </tr>    <tr> 
        <th id="T_74a7695c_f0ad_11e9_ab3d_bc838524d564level0_row3" class="row_heading level0 row3" >3</th> 
        <td id="T_74a7695c_f0ad_11e9_ab3d_bc838524d564row3_col0" class="data row3 col0" >69</td> 
    </tr>    <tr> 
        <th id="T_74a7695c_f0ad_11e9_ab3d_bc838524d564level0_row4" class="row_heading level0 row4" >4</th> 
        <td id="T_74a7695c_f0ad_11e9_ab3d_bc838524d564row4_col0" class="data row4 col0" >11</td> 
    </tr></tbody> 
</table> 




```python
sns.factorplot('Age_band','Survived',data=data,col='Pclass')
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_67_0.png">


Pclass와 상관없이 나이가 줄어들면 생존 확률이 높아지는 것을 볼 수 있습니다.

## Family_Size and Alone


```python
data['Family_Size']=0
data['Family_Size']=data['Parch']+data['SibSp']
data['Alone']=0
data.loc[data.Family_Size==0,'Alone']=1

f,ax=plt.subplots(1,2,figsize=(18,6))
sns.factorplot('Family_Size','Survived',data=data,ax=ax[0])
ax[0].set_title('Family_Size vs Survived')
sns.factorplot('Alone','Survived',data=data,ax=ax[1])
ax[1].set_title('Alone vs Survived')
plt.close(2)
plt.close(3)
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_70_0.png">



```python
sns.factorplot('Alone','Survived',data=data,hue='Sex',col='Pclass')
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_71_0.png">


## Fare_Range


```python
data['Fare_Range']=pd.qcut(data['Fare'],4)
data.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
```




<style  type="text/css" >
    #T_8b4f0a7a_f0ad_11e9_9028_bc838524d564row0_col0 {
            background-color:  #ffff66;
        }    #T_8b4f0a7a_f0ad_11e9_9028_bc838524d564row1_col0 {
            background-color:  #b9dc66;
        }    #T_8b4f0a7a_f0ad_11e9_9028_bc838524d564row2_col0 {
            background-color:  #54aa66;
        }    #T_8b4f0a7a_f0ad_11e9_9028_bc838524d564row3_col0 {
            background-color:  #008066;
        }</style>  
<table id="T_8b4f0a7a_f0ad_11e9_9028_bc838524d564" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >Survived</th> 
    </tr>    <tr> 
        <th class="index_name level0" >Fare_Range</th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_8b4f0a7a_f0ad_11e9_9028_bc838524d564level0_row0" class="row_heading level0 row0" >(-0.001, 7.91]</th> 
        <td id="T_8b4f0a7a_f0ad_11e9_9028_bc838524d564row0_col0" class="data row0 col0" >0.197309</td> 
    </tr>    <tr> 
        <th id="T_8b4f0a7a_f0ad_11e9_9028_bc838524d564level0_row1" class="row_heading level0 row1" >(7.91, 14.454]</th> 
        <td id="T_8b4f0a7a_f0ad_11e9_9028_bc838524d564row1_col0" class="data row1 col0" >0.303571</td> 
    </tr>    <tr> 
        <th id="T_8b4f0a7a_f0ad_11e9_9028_bc838524d564level0_row2" class="row_heading level0 row2" >(14.454, 31.0]</th> 
        <td id="T_8b4f0a7a_f0ad_11e9_9028_bc838524d564row2_col0" class="data row2 col0" >0.454955</td> 
    </tr>    <tr> 
        <th id="T_8b4f0a7a_f0ad_11e9_9028_bc838524d564level0_row3" class="row_heading level0 row3" >(31.0, 512.329]</th> 
        <td id="T_8b4f0a7a_f0ad_11e9_9028_bc838524d564row3_col0" class="data row3 col0" >0.581081</td> 
    </tr></tbody> 
</table> 




```python
data['Fare_cat']=0
data.loc[data['Fare']<=7.91,'Fare_cat']=0
data.loc[(data['Fare']>7.91)&(data['Fare']<=14.454),'Fare_cat']=1
data.loc[(data['Fare']>14.454)&(data['Fare']<=31),'Fare_cat']=2
data.loc[(data['Fare']>31)&(data['Fare']<=513),'Fare_cat']=3
```


```python
sns.factorplot('Fare_cat','Survived',data=data,hue='Sex')
plt.show()
```

<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_75_0.png">


## Converting String Values into Numeric


```python
data['Sex'].replace(['male','female'],[0,1],inplace=True)
data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
```

## Dropping UnNeeded Features


```python
data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)
sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})
fig=plt.gcf()
fig.set_size_inches(18,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
```


<img src = "https://py-tonic.github.io/images/Titanic_files/Titanic_79_0.png">

