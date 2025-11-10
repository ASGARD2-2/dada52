import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# 1)завантаження даних
d=pd.read_csv("lab6 52/insurance.csv")

# 2)вибор потрібних колонок
d=d[['age','region','smoker','expenses']].copy()

# 3)вписування змінних
d['smk']=d['smoker'].map({'yes':1,'no':0})
reg=pd.get_dummies(d['region'],prefix='r',drop_first=True)
d=pd.concat([d.drop(columns=['region','smoker']),reg],axis=1)

# 4)розділення даних
X=d.drop(columns=['expenses'])
y=d['expenses']
X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.25,random_state=42)

# 5)модель
m=LinearRegression()
m.fit(X_tr,y_tr)
y_pr=m.predict(X_te)

# 6)похибки
err=np.abs(y_te-y_pr)
perr=err/y_te*100

# 7)результати
r=X_te.copy()
r['real']=y_te
r['pred']=y_pr
r['perr']=perr

print("Перші 10 результатів:")
print(r.head(10))

# 8)метрики
mae=metrics.mean_absolute_error(y_te,y_pr)
r2=metrics.r2_score(y_te,y_pr)
print(f"\nMAE={mae:.2f}")
print(f"R*R={r2:.4f}")

# 9)гістограма похибок
plt.hist(perr,bins=30,color='skyblue',edgecolor='black')
plt.title("Гістограма відсоткових похибок")
plt.xlabel("Похибка (%)")
plt.ylabel("Кількість")
plt.show()


