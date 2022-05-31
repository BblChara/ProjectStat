import pandas as pd
import numpy as np
from math import *
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
import timedelta


def pooled_standard_error(a, b, unbias=False):
    std1 = a.std(ddof=0) if unbias == False else a.std()
    std2 = b.std(ddof=0) if unbias == False else b.std()
    x = std1 ** 2 / a.count()
    y = std2 ** 2 / b.count()
    return sp.sqrt(x + y)

def z_stat(a, b, unbias=False):
    return (a.mean() - b.mean()) / pooled_standard_error(a, b, unbias)

def z_test(a, b):
    return stats.norm.cdf([z_stat(a, b)])

def check(a, b):
    print('z-статистика:', z_stat(a, b))
    print('p-значение:  ', z_test(a, b))


# Сброс ограничений на количество выводимых рядов
pd.set_option('display.max_rows', None)
# Сброс ограничений на число столбцов
pd.set_option('display.max_columns', None)
# Сброс ограничений на количество символов в записи
pd.set_option('display.max_colwidth', None)

# Cчитывание

BezMono = pd.read_excel('БД без моно.xlsx')
SMono = pd.read_excel('БД с моно.xlsx')
PriznakF = pd.read_excel('Показатель_F.xlsx')
PriznakD = pd.read_excel('Показатель_D.xlsx')

# Ищем максимальный ID среди данных, чтобы затем заполнять NaN

maxID = max(max(BezMono['CaseID']), max(SMono['CaseID']), max(PriznakF['CaseID']), max(PriznakD['CaseID']))


BezMono['CaseID'] = BezMono['CaseID'].replace({np.nan: -1})
SMono['CaseID'] = SMono['CaseID'].replace({np.nan: -1})

# Присваиваем уникальные ID вместо NaN

for t in range(BezMono['CaseID'].shape[0]):
    if BezMono['CaseID'][t] == -1:
        maxID += 1
        BezMono['CaseID'][t] = maxID

for t in range(SMono['CaseID'].shape[0]):
    if SMono['CaseID'][t] == -1:
        maxID += 1
        SMono['CaseID'][t] = maxID

# Сортируем данные по ID и дате для удаления повторных записей (и удаляем их)

BezMono = BezMono.sort_values(['CaseID', 'End'])
SMono = SMono.sort_values(['CaseID', 'End'])

BezMono = BezMono.drop_duplicates(subset=['CaseID'], keep='last')
SMono = SMono.drop_duplicates(subset=['CaseID'], keep='last')

# Создаем таблицу из всех пациентов и сортируем старые по ID для скорости поиска


PriznakF = PriznakF.sort_values(['CaseID'])
PriznakD = PriznakD.sort_values(['CaseID'])
BezMono = BezMono.sort_values(['CaseID'])
SMono = SMono.sort_values(['CaseID'])

# Добавляем пациентам значения F и D и после корректируем возникшие расхождения

BezMono = BezMono.merge(PriznakF[['Результат', 'CaseID']], on=['CaseID'])
BezMono = BezMono.merge(PriznakD[['Результат_D', 'CaseID']], on=['CaseID'])
SMono = SMono.merge(PriznakF[['Результат', 'CaseID']], on=['CaseID'])
SMono = SMono.merge(PriznakD[['Результат_D', 'CaseID']], on=['CaseID'])

BezMono['Результат'] = BezMono['Результат'].replace(
    {'нет пробирки': 0, 'перебрать': 0, '': 0, '283,8227,9': 0, 21123: 0, 17752: 0, 13375: 0, 9912.3: 0, 4662: 0})
SMono['Результат'] = SMono['Результат'].replace({'нет пробирки': 0, 'перебрать': 0, '': 0, '283,8227,9': 0})

BezMono['Результат_D'] = BezMono['Результат_D'].replace(
    {'>3000': 0, '>3000.0': 0, 'более 3000': 0, '>': 0, '': 0, '10,94,84': 0, '10,94,84': 0, '192...97': 0, \
     'наруш. соотношения': 0, 'наруш.соотношения': 0, 'наруш-е соотношения': 0, 'нарушен-е соотношения': 0,
     'нет пробирки': 0, 'перебрать': 0, 'перезабор.мало крови': 0, "сгусток": 0, 'сгусток!!!': 0})
SMono['Результат_D'] = SMono['Результат_D'].replace(
    {'>3000': 0, '>3000.0': 0, 'более 3000': 0, '>': 0, '': 0, '10,94,84': 0, '10,94,84': 0, '192...97': 0,
     'наруш. соотношения': 0, \
     'наруш.соотношения': 0, 'наруш-е соотношения': 0, 'нарушен-е соотношения': 0, 'нет пробирки': 0, 'перебрать': 0,
     'перезабор.мало крови': 0, "сгусток": 0, 'сгусток!!!': 0, 35712: 0})

BezMono['Ther'] = BezMono['Ther'].replace(
    {'ИНФ (крайне тяжелое течение) без ЛП и терапии': 0, \
     'ИНФ (тяжелое течение) без ЛП и терапии': 0,
     'ИНФ (среднетяжелое течение) без ЛП и терапии': 1,\
     'ИНФ (среднетяжелое течение) с ЛП без терапии': 1})
SMono['Ther'] = SMono['Ther'].replace(
    {'ИНФ (крайне  тяжелое течение) с терапией и ЛП': 0,\
     'ИНФ (крайне тяжелое течение) с терапией без ЛП': 0,
     'ИНФ (среднетяжелое течение) с терапией без ЛП': 1, \
     'ИНФ (среднетяжелое течение) с терапией и ЛП': 1,\
     'ИНФ (тяжелое течение) с терапией без ЛП': 0,
     'ИНФ (тяжелое течение) с терапией и ЛП': 0, })

BezMono['Outcome'] = BezMono['Outcome'].replace({'Выписан': 1, 'Умер': 0})
SMono['Outcome'] = SMono['Outcome'].replace({'Выписан': 1, 'Умер': 0})
BezMono['Gender'] = BezMono['Gender'].replace({'м': 1, 'ж': 0})
SMono['Gender'] = SMono['Gender'].replace({'м': 1, 'ж': 0})
BezMono['Результат'] = BezMono['Результат'].replace({np.nan: 1})
SMono['Результат'] = SMono['Результат'].replace({np.nan: 1})
BezMono['Результат_D'] = BezMono['Результат_D'].replace({np.nan: 1})
SMono['Результат_D'] = SMono['Результат_D'].replace({np.nan: 1})


# Проверяем таблицы с показателями на наличие пустых результатов

# print(PriznakF.isnull().sum())

# Оставляем только пациентов  у которых известен исход
BezMono = pd.concat([BezMono.loc[BezMono['Outcome'] == 0], BezMono.loc[BezMono['Outcome'] == 1]])
SMono = pd.concat([SMono.loc[SMono['Outcome'] == 0], SMono.loc[SMono['Outcome'] == 1]])



# Сравниваем процент мужчин и женщин

# fig, ax = plt.subplots(3,2)
#
# labels = 'Женщины', "Мужчины"
# counterFunc = BezMono.apply(
#     lambda x: True if x['Gender'] == 0 else False , axis=1)
# sum = [len(counterFunc[counterFunc == True].index),len(counterFunc[counterFunc == False].index)]
# ax[0][0].pie(sum, labels = labels, autopct = '%1.1f%%')
#
# labels = 'Женщины', "Мужчины"
# counterFunc = SMono.apply(
#     lambda x: True if x['Gender'] == 0 else False , axis=1)
# sum = [len(counterFunc[counterFunc == True].index),len(counterFunc[counterFunc == False].index)]
# ax[0][1].pie(sum, labels = labels, autopct = '%1.1f%%')
# #
# # Сравниваем процент выживаемости
#
# labels = 'Выжил', "Не выжил"
# counterFunc = BezMono.apply(
#    lambda x: True if x['Outcome'] == 1 else False , axis=1)
# sum = [len(counterFunc[counterFunc == True].index), len(counterFunc[counterFunc == False].index)]
# ax[1][0].pie(sum, labels = labels, autopct = '%1.1f%%', colors = ['c','m'])
#
# labels = 'Выжил', "Не выжил"
# counterFunc = SMono.apply(
#    lambda x: True if x['Outcome'] == 1 else False , axis=1)
# sum = [len(counterFunc[counterFunc == True].index), len(counterFunc[counterFunc == False].index)]
# ax[1][1].pie(sum, labels = labels, autopct = '%1.1f%%', colors = ['c','m'])
# #
# # Распределение возрастов
# ax[2][0].hist(BezMono['Age'], bins = 50)
# ax[2][1].hist(SMono['Age'], bins = 50)
# ax[2][0].set_xlabel("  Средний возраст =  " + str(round(BezMono['Age'].mean())))
# ax[2][1].set_xlabel("  Средний возраст =  " + str(round(SMono['Age'].mean())))
# plt.suptitle( 'Без препаратов     /      С препаратами')
# plt.show()
#
# Делаем подгруппы с людьми, у которых общие характеристики
# BezMono = BezMono.assign(Sel = BezMono['Outcome']*0)
# SMono = SMono.assign(Sel = SMono['Outcome']*0)
# BezMono = BezMono.sort_values(['Gender','Age','Результат','Результат_D'])
# SMono = SMono.sort_values(['Gender','Age','Результат','Результат_D'])
#
# for i,row in BezMono.iterrows():
#     stop =0
#     for j,raw in SMono.iterrows():
#         if row['Gender']==raw['Gender'] and abs(row['Age'] - raw['Age']) < 4 \
#                 and abs(row['Результат']-raw['Результат']) <50 and abs(row['Результат_D']-raw['Результат_D'])<50 \
#                 and row['Ther']==raw['Ther'] \
#                 and row['Sel']==0 and raw['Sel']==0:
#             BezMono['Sel'].loc[i]=1
#             SMono['Sel'].loc[j]=1
#             break
#
#
# BezMonoNew1 = BezMono[BezMono['Sel'] ==1]
# SMonoNew1 = SMono[SMono['Sel'] ==1]
# BezMonoNew1.to_csv('new1.csv')
# SMonoNew1.to_csv('new2.csv')
# BezMonoNew1 = pd.read_csv('new1.csv')
# SMonoNew1 = pd.read_csv('new2.csv')

# Оцениваем сопоставимость получившихся групп
# Пол

# check(BezMonoNew1['Gender'],SMonoNew1['Gender'])
# fig, ax = plt.subplots(1,2)
#
# labels = 'Женщины', "Мужчины"
# counterFunc = BezMonoNew1.apply(
#     lambda x: True if x['Gender'] == 0 else False , axis=1)
# sum = [len(counterFunc[counterFunc == True].index),len(counterFunc[counterFunc == False].index)]
# ax[0].pie(sum, labels = labels, autopct = '%1.1f%%')
#
# labels = 'Женщины', "Мужчины"
# counterFunc = SMonoNew1.apply(
#     lambda x: True if x['Gender'] == 0 else False , axis=1)
# sum = [len(counterFunc[counterFunc == True].index),len(counterFunc[counterFunc == False].index)]
# ax[1].pie(sum, labels = labels, autopct = '%1.1f%%')
# plt.suptitle( 'Без препаратов     /      С препаратами')
# plt.show()

# Возраст
# fig, ax = plt.subplots(1,2)
# check(BezMonoNew1['Age'],SMonoNew1['Age'])
# ax[0].hist(BezMonoNew1['Age'], bins = 50)
# ax[1].hist(SMonoNew1['Age'], bins = 50)
# ax[0].set_xlabel("  Средний возраст =  " + str(round(BezMonoNew1['Age'].mean())))
# ax[1].set_xlabel("  Средний возраст =  " + str(round(SMonoNew1['Age'].mean())))
# plt.show()

# Признак F
# check(BezMonoNew1['Результат'],SMonoNew1['Результат'])
# fig, ax = plt.subplots(1,2)
# ax[0].hist(BezMonoNew1['Результат'], bins = 100)
# ax[1].hist(SMonoNew1['Результат'], bins = 100)
# plt.show()
#
# check(BezMonoNew1['Результат_D'],SMonoNew1['Результат_D'])
# fig, ax = plt.subplots(1,2)
# ax[0].hist(BezMonoNew1['Результат_D'], bins = 100)
# ax[1].hist(SMonoNew1['Результат_D'], bins = 100)
# plt.show()

#Тяжесть
# check(BezMonoNew1['Ther'], SMonoNew1['Ther'])
# fig, ax = plt.subplots(1,2)
# labels = 'Тяжелое', "Среднее"
# counterFunc = BezMonoNew1.apply(
#    lambda x: True if x['Ther'] == 1 else False , axis=1)
# sum = [len(counterFunc[counterFunc == True].index), len(counterFunc[counterFunc == False].index)]
# ax[0].pie(sum, labels = labels, autopct = '%1.1f%%', colors = ['c','m'])
#
# labels = 'Тяжелое', "Среднее"
# counterFunc = SMonoNew1.apply(
#    lambda x: True if x['Ther'] == 1 else False , axis=1)
# sum = [len(counterFunc[counterFunc == True].index), len(counterFunc[counterFunc == False].index)]
# ax[1].pie(sum, labels = labels, autopct = '%1.1f%%', colors = ['c','m'])
# plt.show()

# Сравнение групп по признаку выживаемости
# check(BezMonoNew1['Outcome'],SMonoNew1['Outcome'])
# fig, ax = plt.subplots(1,2)
# labels = 'Выжил', "Не выжил"
# counterFunc = BezMonoNew1.apply(
#    lambda x: True if x['Outcome'] == 1 else False , axis=1)
# sum = [len(counterFunc[counterFunc == True].index), len(counterFunc[counterFunc == False].index)]
# ax[0].pie(sum, labels = labels, autopct = '%1.1f%%', colors = ['c','m'])
#
# labels = 'Выжил', "Не выжил"
# counterFunc = SMonoNew1.apply(
#    lambda x: True if x['Outcome'] == 1 else False , axis=1)
# sum = [len(counterFunc[counterFunc == True].index), len(counterFunc[counterFunc == False].index)]
# ax[1].pie(sum, labels = labels, autopct = '%1.1f%%', colors = ['c','m'])
# plt.suptitle( 'Без препаратов     /      С препаратами')
# plt.show()

# Сравнение сроков выздоравления с препаратами и без
# BezMono1 = BezMono[BezMono['Outcome']==1]
# SMono1 = SMono[SMono['Outcome']==1]
# BezMono1['Пребывание'] = (BezMono1['End']-BezMono1['Start']).dt.days
# SMono1['Пребывание'] = (SMono1['End']-SMono1['Start']).dt.days
# check(BezMono1['Пребывание'], SMono1['Пребывание'])
# fig, ax = plt.subplots(1,2)
# ax[0].hist(BezMono1['Пребывание'], bins = 100)
# ax[1].hist(SMono1['Пребывание'], bins = 100)
# plt.show()


# Оцениваем влияение отдельных факторов на исходы
vse = BezMono.append(SMono, ignore_index=True)
temp = vse['Outcome'].astype(float)

# Возраст

# cor = temp.corr(vse['Age'])
# vse.sort_values(['Age'])
# maximum = max(vse['Age'])
# a = []
# for i in range(3,15):
#     new = vse[vse['Age']<(maximum*i/15)]
#     new1 = new[new['Age']>(maximum*(i-1)/15)]
#     a.append(new1['Outcome'].mean())
# pd.DataFrame(np.array([[maximum*i/15 for i in range(3,15)],\
#                        [s*100 for s in a]]).T).plot.scatter(0, 1, s=50, grid=True)
# plt.xlabel('corr = ' +str(cor))
# plt.show()

# Факт вакцинации
# vse['Vacin']= vse['Vacin'].replace({np.nan:-1, 'Нет':0, 'Спутник V':1, 'Эпиваккорона':1, 'Ковивак':1,'Спутник Лайт':1,'Phizer':1})
# cor =temp.corr(vse['Vacin'])
# count1 = 0
# count2 = 0
# ish1 = 0
# ish2 = 0
#
# vsevac = vse[vse['Vacin']==1]
# vsebezvac = vse[vse['Vacin']==0]
#
# fig, ax = plt.subplots(1,2)
# labels = 'Выжил', "Не выжил"
# counterFunc = vsevac.apply(
#    lambda x: True if x['Outcome'] == 1 else False , axis=1)
# sum = [len(counterFunc[counterFunc == True].index), len(counterFunc[counterFunc == False].index)]
# ax[0].pie(sum, labels = labels, autopct = '%1.1f%%', colors = ['c','m'])
#
# counterFunc1 = vsebezvac.apply(
#    lambda x: True if x['Outcome'] == 1 else False , axis=1)
# sum = [len(counterFunc1[counterFunc1 == True].index), len(counterFunc1[counterFunc1 == False].index)]
#
# ax[1].pie(sum, labels = labels, autopct = '%1.1f%%', colors = ['c','m'])
# plt.suptitle( 'Без Вакцины     /      С Вакциной')
# plt.xlabel('corr = ' +str(cor))
# plt.show()

# Пол
# vsem = vse[vse['Gender']==1]
# vsew = vse[vse['Gender']==0]
# cor = temp.corr(vse['Gender'])
# fig, ax = plt.subplots(1,2)
# labels = 'Выжил', "Не выжил"
# counterFunc = vsem.apply(
#    lambda x: True if x['Outcome'] == 1 else False , axis=1)
# sum = [len(counterFunc[counterFunc == True].index), len(counterFunc[counterFunc == False].index)]
# ax[0].pie(sum, labels = labels, autopct = '%1.1f%%', colors = ['c','m'])
#
# counterFunc1 = vsew.apply(
#    lambda x: True if x['Outcome'] == 1 else False , axis=1)
# sum = [len(counterFunc1[counterFunc1 == True].index), len(counterFunc1[counterFunc1 == False].index)]
#
# ax[1].pie(sum, labels = labels, autopct = '%1.1f%%', colors = ['c','m'])
# plt.suptitle( 'Мужчины     /      Женщины')
# plt.xlabel('corr = ' +str(cor))
#
# plt.show()

# Тяжесть

# vses = vse[vse['Ther']==1]
# vset = vse[vse['Ther']==0]
# cor = temp.corr(vse['Ther'])
# fig, ax = plt.subplots(1,2)
# labels = 'Выжил', "Не выжил"
# counterFunc = vses.apply(
#    lambda x: True if x['Outcome'] == 1 else False , axis=1)
# sum = [len(counterFunc[counterFunc == True].index), len(counterFunc[counterFunc == False].index)]
# ax[0].pie(sum, labels = labels, autopct = '%1.1f%%', colors = ['c','m'])
#
# counterFunc1 = vset.apply(
#    lambda x: True if x['Outcome'] == 1 else False , axis=1)
# sum = [len(counterFunc1[counterFunc1 == True].index), len(counterFunc1[counterFunc1 == False].index)]
#
# ax[1].pie(sum, labels = labels, autopct = '%1.1f%%', colors = ['c','m'])
# plt.suptitle( 'Средней тяжести     /      Тяжелые и крайне тяжелые')
# plt.xlabel('corr = ' +str(cor))
#
# plt.show()

# Признак F
# cor = temp.corr(vse['Результат'])
# vse.sort_values(['Результат'])
# a = []
# maximum = max(vse['Результат'])
# for i in range(3,100):
#     new = vse[vse['Результат']<(maximum*i/100)]
#     new1 = new[new['Результат']>(maximum*(i-1)/100)]
#     a.append(new1['Outcome'].mean())
# pd.DataFrame(np.array([[maximum*i/100 for i in range(3,100)],\
#                        [s*100 for s in a]]).T).plot.scatter(0, 1, s=10, grid=True)
# plt.xlabel('corr = ' +str(cor))
# plt.show()

# Признак D
cor = temp.corr(vse['Результат_D'])
vse.sort_values(['Результат_D'])
a = []
maximum = 3000
for i in range(3,50):
    new = vse[vse['Результат_D']<(maximum*i/50)]
    new1 = new[new['Результат_D']>(maximum*(i-1)/50)]
    a.append(new1['Outcome'].mean())
pd.DataFrame(np.array([[maximum*i/50 for i in range(3,50)],\
                       [s*100 for s in a]]).T).plot.scatter(0, 1, s=10, grid=True)
plt.xlabel('corr = ' +str(cor))
plt.show()
