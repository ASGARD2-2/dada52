import pandas as pd
import matplotlib.pyplot as plt

#1)завантаження цсв
d=pd.read_csv("lab5/titanic.csv")
print(" Інформація про дані")
print(d.info())

#2)перші 5 та останні 10
print("\n Перші 5")
print(d.head())
print("\n Останні 10")
print(d.tail(10))

#3)вижили>30
s30=d[(d["Age"]>30)&(d["Survived"]==1)]
print("\n Вижили старше 30")
print(s30)

#4)з братами і сестрами
sib=d[d["SibSp"]>0]
print("\n З братами/сестрами")
print(sib)

#5)діаграма
def grp(a):
    if pd.isna(a):return None
    elif a<14:return "Діти"
    elif a<18:return "Підлітки"
    elif a<30:return "Молодь"
    elif a<60:return "Працездатні"
    else:return "Пенсіонери"

#вибор виживших і додавання вікових груп
s=d[d["Survived"]==1].copy()
s["Grp"]=s["Age"].apply(grp)

#підрахунок кількості 
c=s["Grp"].value_counts()

#побудова діаграми
plt.bar(c.index,c.values)
plt.xlabel("Категорії віку")
plt.ylabel("Кількість виживших")
plt.title("Розподіл вікових груп серед тих, хто вижив")
plt.show()
