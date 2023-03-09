# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:07:16 2022

@author: Gabriel Ospina
"""
# from sklearn import datasets
import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import plotly.express as px

"""Procesamiento Jornada Diurna"""
cwd = os.path.abspath('') 
files = os.listdir(cwd) 
df = pd.DataFrame()
for file in files:
    if file.endswith('.xlsx'):
        df = df.append(pd.read_excel(file, sheet_name='DIURNO'), ignore_index=True) #Reconocimiento de archivos de excel
df.head()
df.info()

# df['Date'] = df['Date'].str.replace(r'\D', '00:00:00')
# EMRI=df.iloc[:,0]
# EMRIS=pd.DataFrame(EMRI)
# LD=df.iloc[:,12]
print('Row count is:',df.shape[1])

f1_events=pd.DataFrame()
f2_events=pd.DataFrame()
f8_events=pd.DataFrame()
f10_events=pd.DataFrame()
f18_events=pd.DataFrame()
f28_events=pd.DataFrame()
f33_events=pd.DataFrame()

"""Creacion de Matriz con columna de 10^(SEL/10)"""
if df.shape[1]==29:
    df['SEL OACI POT']= 10**(df['SEL OACI']/10)
    df['Date'] = pd.to_datetime(df['Date'])#Convierte el dato de fecha en formato de Pandas
    f1_events=df.groupby(["NMT"]).get_group("F001")
    f2_events=df.groupby(["NMT"]).get_group("F002")
    f3_events=df.groupby(["NMT"]).get_group("F003")
    f4_events=df.groupby(["NMT"]).get_group("F004")
    f5_events=df.groupby(["NMT"]).get_group("F005")
    f7_events=df.groupby(["NMT"]).get_group("F007")
    f8_events=df.groupby(["NMT"]).get_group("F008")
    f10_events=df.groupby(["NMT"]).get_group("F010")
    f11_events=df.groupby(["NMT"]).get_group("F011")
    f13_events=df.groupby(["NMT"]).get_group("F013")
    f15_events=df.groupby(["NMT"]).get_group("F015")
    f17_events=df.groupby(["NMT"]).get_group("F017")
    f18_events=df.groupby(["NMT"]).get_group("F018")
    f19_events=df.groupby(["NMT"]).get_group("F019")
    f20_events=df.groupby(["NMT"]).get_group("F020")
    f21_events=df.groupby(["NMT"]).get_group("F021")
    f23_events=df.groupby(["NMT"]).get_group("F023")
    f24_events=df.groupby(["NMT"]).get_group("F024")
    f25_events=df.groupby(["NMT"]).get_group("F025")
    f27_events=df.groupby(["NMT"]).get_group("F027")
    f28_events=df.groupby(["NMT"]).get_group("F028")
    f29_events=df.groupby(["NMT"]).get_group("F029")
    f30_events=df.groupby(["NMT"]).get_group("F030")
    f32_events=df.groupby(["NMT"]).get_group("F032")
    f33_events=df.groupby(["NMT"]).get_group("F033")
    f34_events=df.groupby(["NMT"]).get_group("F034")
print(df)

"""Creacion de Dataframe por localidades y eventos de ruido"""
"""En Proceso grafico de violines o distribucion de eventos de ruido"""
engativa_events=pd.DataFrame()
engativa_event=[f1_events,f2_events,f8_events,f10_events,f18_events,f28_events,f33_events]
engativa_events=pd.concat(engativa_event)
sns.kdeplot(data=engativa_events, x="SEL OACI", y="Event duration", hue="NMT", levels=5, thresh=.2)

"""Creacion de 2da_Matriz con datos agrupados y nivel por hora"""
"""CÁLCULO NIVELES POR PERFIL HORARIO"""
df2 = df.groupby(["Date","NMT","Hora"], dropna=False)['SEL OACI POT'].sum().reset_index()
df2['LEVEL_H'] = 10*np.log10((1/3600)*df2['SEL OACI POT'])
print(df2)

"""Creacion de 3ra_Matriz con Callsing de SEL maximo"""
#df3 = df.groupby(["Date","NMT","Hora"])['SEL OACI','FIS Aircraft type','Callsign'].max().reset_index()
df3 = df.groupby(["Date","NMT","Hora"])['SEL OACI','Callsign'].max().reset_index()#falta identificar columna de 'FIS Aircraft type'
df3['CALLSIGN MAX'] = df3['Callsign']
print(df3)

"""Creacion de Final_Matriz con datos maximos, nivel por hora de 2da_Matriz y Callsing de SEL Max por hora"""
df_final = df.groupby(["Date","NMT","Hora"])['Leq max 1sec','Leq','Event duration','Leq max elementary','SEL','PNL max','SEL OACI','LEQ OACI','EPNL OACI'].max().reset_index()
df_final['LEVEL_H']=df2['LEVEL_H']
df_final['CALLSING MAX']=df3['Callsign']
df_final['LEVEL_H']
print(df_final)
'NMT' in df_final

"""Calculo nivel diario Ln"""
df_final['LEVEL_H POT']= 10**(df_final['LEVEL_H']/10)
df_ld=df_final.groupby(["Date","NMT"])['LEVEL_H POT'].sum().reset_index()
df_ld['LEVEL_LD']= 10*np.log10((1/14)*df_ld['LEVEL_H POT'])
del df_ld['LEVEL_H POT']#Borra una columna especifica de un DataFrame
#df_ld.insert(5,'copia LHH',df_final['LEVEL_H'])#.insert siver para inserta una columna en un DataFrame especifico
#df_ld['LEVEL_H upper']=df_ld['LEVEL_LD'][:2]#creea una nueva columna con el numero de filas elegidas ej: [:2]
df_ld
#Ld=df_ld.pop('LEVEL_LD')#Transfiere una columnda de un DataFrame especifico a un vector
df_ld.columns
df_ld.index#Verifica cual es el index
df_ld=df_ld.set_index(['Date'])#Configrar el index por una variable especifica
'Date' in df_ld#Verifica si el valor se ecuentra en el DataFrame

"""Calculo de niveles por modelo de aeronave"""


aircraft_type=df["FIS Aircraft type"].value_counts()#Cuenta el numero de valores por item de modelo de aeronave
niv_air_type_f1=f1_events.groupby(['Date','FIS Aircraft type','Direction']).sum()#Suma los niveles SEL OACI POT para cada modelo de aeronave
niv_air_type_f1['LEVEL_Aircraft_type'] = 10*np.log10((1/(3600*14))*niv_air_type_f1['SEL OACI POT'])#Calcula el nivel equivalente diario para cada modelo de aeronave
type_air_dir = f1_events.groupby(['FIS Aircraft type','Direction'])['SEL OACI'].apply(list).reset_index(name='SEL OACI')

z=type_air_dir.iloc[1,2]
air_list =list(f1_events["FIS Aircraft type"])
print(air_list)
index_list = aircraft_type.index.tolist()
print(index_list)
list_pd = pd.DataFrame (index_list, columns = ['FIS Aircraft type'])

"""NIVEL DE APORTE DIARIO POR MODELO DE AEROANVE Y TIPO DE OPERACION"""

num_eve_all=df["FIS Aircraft type"].value_counts()#Cuenta el numero de valores por item de modelo de aeronave
niv_air_type_perday=df.groupby(['Date','NMT','FIS Aircraft type','Direction']).sum()#Suma los niveles SEL OACI POT para cada modelo de aeronave, estacion, direccion y dia
niv_air_type_permonth=df.groupby(['NMT','FIS Aircraft type','Direction']).sum()#Suma los niveles SEL OACI POT para cada modelo de aeronave, estacion, direccion mensual
niv_air_type_perday['LEVEL_Aircraft_type'] = 10*np.log10((1/(3600*14))*niv_air_type_perday['SEL OACI POT'])#Calcula el nivel equivalente diario para cada modelp

niv_air_type_permonth['LEVEL_Aircraft_type'] = 10*np.log10((1/(3600*14*31))*niv_air_type_permonth['SEL OACI POT'])#Calcula el nivel equivalente mensual para cada modelp
niv_air_type_perday['LEVEL_Aircraft_type_POT']= 10**(niv_air_type_perday['LEVEL_Aircraft_type']/10)
niv_air_type_perday=niv_air_type_perday.reset_index()
niv_air_type_permonth['LEVEL_Aircraft_type_POT']= 10**(niv_air_type_permonth['LEVEL_Aircraft_type']/10)
niv_air_type_permonth=niv_air_type_permonth.reset_index()

"""NIVEL MAXIMO DE APORTE DIARIO POR MODELO DE AERONAVE Y TIPO DE OPERACION"""

top_niv_air_perday=niv_air_type_perday.copy()
top_niv_air_perday=top_niv_air_perday
top_niv_air_perday_max=top_niv_air_perday.groupby(["Date"])['LEVEL_Aircraft_type'].transform(max) == top_niv_air_perday['LEVEL_Aircraft_type']
top_niv_air_perday_max=top_niv_air_perday[top_niv_air_perday_max]
top_niv_air_perday_max=top_niv_air_perday_max.round(1)

"""VERIFICACION DE NIVEL DIARIO Y MENSUAL POR ESTACION"""

niv_air_perday=niv_air_type_perday.groupby(['Date','NMT']).sum()#Suma los niveles SEL OACI POT para cada modelo de aeronave, estacion, direccion y dia
niv_air_permonth=niv_air_type_permonth.groupby('NMT').sum()
num_aerolineas = niv_air_type_perday['FIS Aircraft type'].nunique()
niv_air_perday['LEVEL_Aircraft_day'] = 10*np.log10(niv_air_perday['LEVEL_Aircraft_type_POT'])
niv_air_perday=niv_air_perday.reset_index()
niv_air_permonth['LEVEL_Aircraft_month'] = 10*np.log10(niv_air_permonth['LEVEL_Aircraft_type_POT'])
niv_air_permonth=niv_air_permonth.reset_index()

index_list_all = num_eve_all.index.tolist()
list_aircraft=pd.DataFrame(index_list_all)
for c in index_list_all:
    exec('{} = pd.DataFrame()'.format(c))
    UNK = (df.loc[df['FIS Aircraft type'] == 'UNK'])
    UNK_DIR = UNK.groupby('Direction')  # Agrupa los datos en tipo NoneType
    # UNK_DEP = UNK_DIR.get_group('DEP')  # Agrupa los datos por estacion
    # UNK_DEP = pd.DataFrame(UNK_DEP, columns=['SEL OACI'])  # Agrupa niveles diurnos en tipo DataFrame
    # UNK_DEP = UNK_DEP.rename({'SEL OACI': 'UNK_DEP'}, axis=1)
    # UNK_ARR = UNK_DIR.get_group('ARR')
    # UNK_ARR = pd.DataFrame(UNK_ARR, columns=['SEL OACI'])
    # UNK_ARR = UNK_ARR.rename({'SEL OACI': 'UNK_ARR'}, axis=1)
    # UNK_UNK = UNK_DIR.get_group('UNK')
    # UNK_UNK = pd.DataFrame(UNK_UNK, columns=['SEL OACI'])
    # UNK_UNK = UNK_UNK.rename({'SEL OACI': 'UNK_UNK'}, axis=1)
    A320 = (df.loc[df['FIS Aircraft type'] == 'A320'])
    A320_DIR = A320.groupby('Direction')
    A320_DEP = A320_DIR.get_group('DEP')
    A320_DEP = pd.DataFrame(A320_DEP, columns=['SEL OACI'])
    A320_DEP = A320_DEP.rename({'SEL OACI': 'A320_DEP'}, axis=1)
    A320_ARR = A320_DIR.get_group('ARR')
    A320_ARR = pd.DataFrame(A320_ARR, columns=['SEL OACI'])
    A320_ARR = A320_ARR.rename({'SEL OACI': 'A320_ARR'}, axis=1)
    # A320_UNK = A320_DIR.get_group('UNK')
    # A320_UNK = pd.DataFrame(A320_UNK, columns=['SEL OACI'])
    # A320_UNK = A320_UNK.rename({'SEL OACI': 'A320_UNK'}, axis=1)
    A319 = (df.loc[df['FIS Aircraft type'] == 'A319'])
    A319_DIR = A319.groupby('Direction')
    A319_DEP = A319_DIR.get_group('DEP')
    A319_DEP = pd.DataFrame(A319_DEP, columns=['SEL OACI'])
    A319_DEP = A319_DEP.rename({'SEL OACI': 'A319_DEP'}, axis=1)
    A319_ARR = A319_DIR.get_group('ARR')
    A319_ARR = pd.DataFrame(A319_ARR, columns=['SEL OACI'])
    A319_ARR = A319_ARR.rename({'SEL OACI': 'A319_ARR'}, axis=1)
    # A319_UNK = A319_DIR.get_group('UNK')
    # A319_UNK = pd.DataFrame(A319_UNK, columns=['SEL OACI'])
    # A319_UNK = A319_UNK.rename({'SEL OACI': 'A319_UNK'}, axis=1)
    B738 = (df.loc[df['FIS Aircraft type'] == 'B738'])
    B738_DIR = B738.groupby('Direction')
    B738_DEP = B738_DIR.get_group('DEP')
    B738_DEP = pd.DataFrame(B738_DEP, columns=['SEL OACI'])
    B738_DEP = B738_DEP.rename({'SEL OACI': 'B738_DEP'}, axis=1)
    B738_ARR = B738_DIR.get_group('ARR')
    B738_ARR = pd.DataFrame(B738_ARR, columns=['SEL OACI'])
    B738_ARR = B738_ARR.rename({'SEL OACI': 'B738_ARR'}, axis=1)
    # B738_UNK = B738_DIR.get_group('UNK')
    # B738_UNK = pd.DataFrame(B738_UNK, columns=['SEL OACI'])
    # B738_UNK = B738_UNK.rename({'SEL OACI': 'B738_UNK'}, axis=1)
    A20N = (df.loc[df['FIS Aircraft type'] == 'A20N'])
    A20N_DIR = A20N.groupby('Direction')
    A20N_DEP = A20N_DIR.get_group('DEP')
    A20N_DEP = pd.DataFrame(A20N_DEP, columns=['SEL OACI'])
    A20N_DEP = A20N_DEP.rename({'SEL OACI': 'A20N_DEP'}, axis=1)
    A20N_ARR = A20N_DIR.get_group('ARR')
    A20N_ARR = pd.DataFrame(A20N_ARR, columns=['SEL OACI'])
    A20N_ARR = A20N_ARR.rename({'SEL OACI': 'A20N_ARR'}, axis=1)
    # A20N_UNK = A20N_DIR.get_group('UNK')
    # A20N_UNK = pd.DataFrame(A20N_UNK, columns=['SEL OACI'])
    # A20N_UNK = A20N_UNK.rename({'SEL OACI': 'A20N_UNK'}, axis=1)
    B722 = (df.loc[df['FIS Aircraft type'] == 'B722'])
    B722_DIR = B722.groupby('Direction')
    B722_DEP = B722_DIR.get_group('DEP')
    B722_DEP = pd.DataFrame(B722_DEP, columns=['SEL OACI'])
    B722_DEP = B722_DEP.rename({'SEL OACI': 'B722_DEP'}, axis=1)
    B722_ARR = B722_DIR.get_group('ARR')
    B722_ARR = pd.DataFrame(B722_ARR, columns=['SEL OACI'])
    B722_ARR = B722_ARR.rename({'SEL OACI': 'B722_ARR'}, axis=1)
    # B722_UNK = B722_DIR.get_group('UNK')
    # B722_UNK = pd.DataFrame(B722_UNK, columns=['SEL OACI'])
    # B722_UNK = B722_UNK.rename({'SEL OACI': 'B722_UNK'}, axis=1)
    A332 = (df.loc[df['FIS Aircraft type'] == 'A332'])
    A332_DIR = A332.groupby('Direction')
    A332_DEP = A332_DIR.get_group('DEP')
    A332_DEP = pd.DataFrame(A332_DEP, columns=['SEL OACI'])
    A332_DEP = A332_DEP.rename({'SEL OACI': 'A332_DEP'}, axis=1)
    A332_ARR = A332_DIR.get_group('ARR')
    A332_ARR = pd.DataFrame(A332_ARR, columns=['SEL OACI'])
    A332_ARR = A332_ARR.rename({'SEL OACI': 'A332_ARR'}, axis=1)
    # A332_UNK = A332_DIR.get_group('UNK')
    # A332_UNK = pd.DataFrame(A332_UNK, columns=['SEL OACI'])
    # A332_UNK = A332_UNK.rename({'SEL OACI': 'A332_UNK'}, axis=1)
    B763 = (df.loc[df['FIS Aircraft type'] == 'B763'])
    B763_DIR = B763.groupby('Direction')
    B763_DEP = B763_DIR.get_group('DEP')
    B763_DEP = pd.DataFrame(B763_DEP, columns=['SEL OACI'])
    B763_DEP = B763_DEP.rename({'SEL OACI': 'B763_DEP'}, axis=1)
    B763_ARR = B763_DIR.get_group('ARR')
    B763_ARR = pd.DataFrame(B763_ARR, columns=['SEL OACI'])
    B763_ARR = B763_ARR.rename({'SEL OACI': 'B763_ARR'}, axis=1)
    # B763_UNK = B763_DIR.get_group('UNK')
    # B763_UNK = pd.DataFrame(B763_UNK, columns=['SEL OACI'])
    # B763_UNK = B763_UNK.rename({'SEL OACI': 'B763_UNK'}, axis=1)
    B788 = (df.loc[df['FIS Aircraft type'] == 'B788'])
    B788_DIR = B788.groupby('Direction')
    B788_DEP = B788_DIR.get_group('DEP')
    B788_DEP = pd.DataFrame(B788_DEP, columns=['SEL OACI'])
    B788_DEP = B788_DEP.rename({'SEL OACI': 'B788_DEP'}, axis=1)
    B788_ARR = B788_DIR.get_group('ARR')
    B788_ARR = pd.DataFrame(B788_ARR, columns=['SEL OACI'])
    B788_ARR = B788_ARR.rename({'SEL OACI': 'B788_ARR'}, axis=1)
    # B788_UNK = B788_DIR.get_group('UNK')
    # B788_UNK = pd.DataFrame(B788_UNK, columns=['SEL OACI'])
    # B788_UNK = B788_UNK.rename({'SEL OACI': 'B788_UNK'}, axis=1)
    B190 = (df.loc[df['FIS Aircraft type'] == 'B190'])
    B190_DIR = B190.groupby('Direction')
    B190_DEP = B190_DIR.get_group('DEP')
    B190_DEP = pd.DataFrame(B190_DEP, columns=['SEL OACI'])
    B190_DEP = B190_DEP.rename({'SEL OACI': 'B190_DEP'}, axis=1)
    # B190_ARR = B190_DIR.get_group('ARR')
    # B190_ARR = pd.DataFrame(B190_ARR, columns=['SEL OACI'])
    # B190_ARR = B190_ARR.rename({'SEL OACI': 'B190_ARR'}, axis=1)
    # B190_UNK = B190_DIR.get_group('UNK')
    # B190_UNK = pd.DataFrame(B190_UNK, columns=['SEL OACI'])
    # B190_UNK = B190_UNK.rename({'SEL OACI': 'B190_UNK'}, axis=1)
    AT45 = (df.loc[df['FIS Aircraft type'] == 'AT45'])
    AT45_DIR = AT45.groupby('Direction')
    AT45_DEP = AT45_DIR.get_group('DEP')
    AT45_DEP = pd.DataFrame(AT45_DEP, columns=['SEL OACI'])
    AT45_DEP = AT45_DEP.rename({'SEL OACI': 'AT45_DEP'}, axis=1)
    AT45_ARR = AT45_DIR.get_group('ARR')
    AT45_ARR = pd.DataFrame(AT45_ARR, columns=['SEL OACI'])
    AT45_ARR = AT45_ARR.rename({'SEL OACI': 'AT45_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    B737 = (df.loc[df['FIS Aircraft type'] == 'B737'])
    B737_DIR = B737.groupby('Direction')
    B737_DEP = B737_DIR.get_group('DEP')
    B737_DEP = pd.DataFrame(AT45_DEP, columns=['SEL OACI'])
    B737_DEP = B737_DEP.rename({'SEL OACI': 'B737_DEP'}, axis=1)
    B737_ARR = B737_DIR.get_group('ARR')
    B737_ARR = pd.DataFrame(B737_ARR, columns=['SEL OACI'])
    B737_ARR = B737_ARR.rename({'SEL OACI': 'B737_ARR'}, axis=1)
    # B737_UNK = B737_DIR.get_group('UNK')
    # B737_UNK = pd.DataFrame(B737_UNK, columns=['SEL OACI'])
    # B737_UNK = B737_UNK.rename({'SEL OACI': 'B737_UNK'}, axis=1)
    A321 = (df.loc[df['FIS Aircraft type'] == 'A321'])
    A321_DIR = A321.groupby('Direction')
    # A321_DEP = A321_DIR.get_group('DEP')
    # A321_DEP = pd.DataFrame(A321_DEP, columns=['SEL OACI'])
    # A321_DEP = A321_DEP.rename({'SEL OACI': 'A321_DEP'}, axis=1)
    # A321_ARR = A321_DIR.get_group('ARR')
    # A321_ARR = pd.DataFrame(A321_ARR, columns=['SEL OACI'])
    # A321_ARR = A321_ARR.rename({'SEL OACI': 'A321_ARR'}, axis=1)
    # A321_UNK = A321_DIR.get_group('UNK')
    # A321_UNK = pd.DataFrame(A321_UNK, columns=['SEL OACI'])
    # A321_UNK = A321_UNK.rename({'SEL OACI': 'A321_UNK'}, axis=1)
    B734 = (df.loc[df['FIS Aircraft type'] == 'B734'])
    B734_DIR = B734.groupby('Direction')
    B734_DEP = B734_DIR.get_group('DEP')
    B734_DEP = pd.DataFrame(B734_DEP, columns=['SEL OACI'])
    B734_DEP = B734_DEP.rename({'SEL OACI': 'B734_DEP'}, axis=1)
    B734_ARR = B734_DIR.get_group('ARR')
    B734_ARR = pd.DataFrame(B734_ARR, columns=['SEL OACI'])
    B734_ARR = B734_ARR.rename({'SEL OACI': 'B734_ARR'}, axis=1)
    # B734_UNK = B734_DIR.get_group('UNK')
    # B734_UNK = pd.DataFrame(B734_UNK, columns=['SEL OACI'])
    # B734_UNK = B734_UNK.rename({'SEL OACI': 'B734_UNK'}, axis=1)
    B38M = (df.loc[df['FIS Aircraft type'] == 'B38M'])
    B38M_DIR = B38M.groupby('Direction')
    B38M_DEP = B38M_DIR.get_group('DEP')
    B38M_DEP = pd.DataFrame(B38M_DEP, columns=['SEL OACI'])
    B38M_DEP = B38M_DEP.rename({'SEL OACI': 'B38M_DEP'}, axis=1)
    # B38M_ARR = B38M_ARR.get_group('ARR')
    # B38M_ARR = pd.DataFrame(B38M_ARR, columns=['SEL OACI'])
    # B38M_ARR = B38M_ARR.rename({'SEL OACI': 'B38M_ARR'}, axis=1)
    # B38M_UNK = B38M_DIR.get_group('UNK')
    # B38M_UNK = pd.DataFrame(B38M_UNK, columns=['SEL OACI'])
    # B38M_UNK = B38M_UNK.rename({'SEL OACI': 'B38M_UNK'}, axis=1)
    B744 = (df.loc[df['FIS Aircraft type'] == 'B744'])
    B744_DIR = B744.groupby('Direction')
    B744_DEP = B744_DIR.get_group('DEP')
    B744_DEP = pd.DataFrame(B744_DEP, columns=['SEL OACI'])
    B744_DEP = B744_DEP.rename({'SEL OACI': 'B744_DEP'}, axis=1)
    B744_ARR = B744_DIR.get_group('ARR')
    B744_ARR = pd.DataFrame(B744_ARR, columns=['SEL OACI'])
    B744_ARR = B744_ARR.rename({'SEL OACI': 'B744_ARR'}, axis=1)
    # B744_UNK = B744_DIR.get_group('UNK')
    # B744_UNK = pd.DataFrame(B744_UNK, columns=['SEL OACI'])
    # B744_UNK = B744_UNK.rename({'SEL OACI': 'B744_UNK'}, axis=1)
    B789 = (df.loc[df['FIS Aircraft type'] == 'B789'])
    B789_DIR = B789.groupby('Direction')
    B789_DEP = B789_DIR.get_group('DEP')
    B789_DEP = pd.DataFrame(B789_DEP, columns=['SEL OACI'])
    B789_DEP = B789_DEP.rename({'SEL OACI': 'B789_DEP'}, axis=1)
    B789_ARR = B789_DIR.get_group('ARR')
    B789_ARR = pd.DataFrame(B789_ARR, columns=['SEL OACI'])
    B789_ARR = B789_ARR.rename({'SEL OACI': 'B789_ARR'}, axis=1)
    # B789_UNK = B789_DIR.get_group('UNK')
    # B789_UNK = pd.DataFrame(B789_UNK, columns=['SEL OACI'])
    # B789_UNK = B789_UNK.rename({'SEL OACI': 'B789_UNK'}, axis=1)
    A359 = (df.loc[df['FIS Aircraft type'] == 'A359'])
    A359_DIR = A359.groupby('Direction')
    A359_DEP = A359_DIR.get_group('DEP')
    A359_DEP = pd.DataFrame(A359_DEP, columns=['SEL OACI'])
    A359_DEP = A359_DEP.rename({'SEL OACI': 'A359_DEP'}, axis=1)
    # A359_ARR = A359_DIR.get_group('ARR')
    # A359_ARR = pd.DataFrame(A359_ARR, columns=['SEL OACI'])
    # A359_ARR = A359_ARR.rename({'SEL OACI': 'A359_ARR'}, axis=1)
    # A359_UNK = A359_DIR.get_group('UNK')
    # A359_UNK = pd.DataFrame(A359_UNK, columns=['SEL OACI'])
    # A359_UNK = A359_UNK.rename({'SEL OACI': 'A359_UNK'}, axis=1)
    A306 = (df.loc[df['FIS Aircraft type'] == 'A306'])
    A306_DIR = A306.groupby('Direction')
    A306_DEP = A306_DIR.get_group('DEP')
    A306_DEP = pd.DataFrame(A306_DEP, columns=['SEL OACI'])
    A306_DEP = A306_DEP.rename({'SEL OACI': 'A306_DEP'}, axis=1)
    A306_ARR = A306_DIR.get_group('ARR')
    A306_ARR = pd.DataFrame(A306_ARR, columns=['SEL OACI'])
    A306_ARR = A306_ARR.rename({'SEL OACI': 'A306_ARR'}, axis=1)
    # A306_UNK = A306_DIR.get_group('UNK')
    # A306_UNK = pd.DataFrame(A306_UNK, columns=['SEL OACI'])
    # A306_UNK = A306_UNK.rename({'SEL OACI': 'A306_UNK'}, axis=1)
    AT76 = (df.loc[df['FIS Aircraft type'] == 'AT76'])
    AT76_DIR = AT76.groupby('Direction')
    AT76_DEP = AT76_DIR.get_group('DEP')
    AT76_DEP = pd.DataFrame(AT76_DEP, columns=['SEL OACI'])
    AT76_DEP = AT76_DEP.rename({'SEL OACI': 'AT76_DEP'}, axis=1)
    AT76_ARR = AT76_DIR.get_group('ARR')
    AT76_ARR = pd.DataFrame(AT76_ARR, columns=['SEL OACI'])
    AT76_ARR = AT76_ARR.rename({'SEL OACI': 'AT76_ARR'}, axis=1)
    # AT76_UNK = AT76_DIR.get_group('UNK')
    # AT76_UNK = pd.DataFrame(AT76_UNK, columns=['SEL OACI'])
    # AT76_UNK = AT76_UNK.rename({'SEL OACI': 'AT76_UNK'}, axis=1)
    A333 = (df.loc[df['FIS Aircraft type'] == 'A333'])
    A333_DIR = A333.groupby('Direction')
    # A333_DEP = A333_DIR.get_group('DEP')
    # A333_DEP = pd.DataFrame(A333_DEP, columns=['SEL OACI'])
    # A333_DEP = A333_DEP.rename({'SEL OACI': 'A333_DEP'}, axis=1)
    # A333_ARR = A333_DIR.get_group('ARR')
    # A333_ARR = pd.DataFrame(A333_ARR, columns=['SEL OACI'])
    # A333_ARR = A333_ARR.rename({'SEL OACI': 'A333_ARR'}, axis=1)
    # A333_UNK = A333_DIR.get_group('UNK')
    # A333_UNK = pd.DataFrame(A333_UNK, columns=['SEL OACI'])
    # A333_UNK = A333_UNK.rename({'SEL OACI': 'A333_UNK'}, axis=1)
    AT75 = (df.loc[df['FIS Aircraft type'] == 'AT75'])
    AT75_DIR = AT75.groupby('Direction')
    AT75_DEP = AT75_DIR.get_group('DEP')
    AT75_DEP = pd.DataFrame(AT75_DEP, columns=['SEL OACI'])
    AT75_DEP = AT75_DEP.rename({'SEL OACI': 'AT75_DEP'}, axis=1)
    AT75_ARR = AT75_DIR.get_group('ARR')
    AT75_ARR = pd.DataFrame(AT75_ARR, columns=['SEL OACI'])
    AT75_ARR = AT75_ARR.rename({'SEL OACI': 'AT75_ARR'}, axis=1)
    # AT75_UNK = AT75_DIR.get_group('UNK')
    # AT75_UNK = pd.DataFrame(AT75_UNK, columns=['SEL OACI'])
    # AT75_UNK = AT75_UNK.rename({'SEL OACI': 'AT75_UNK'}, axis=1)
    E145 = (df.loc[df['FIS Aircraft type'] == 'E145'])
    E145_DIR = E145.groupby('Direction')
    E145_DEP = E145_DIR.get_group('DEP')
    E145_DEP = pd.DataFrame(E145_DEP, columns=['SEL OACI'])
    E145_DEP = E145_DEP.rename({'SEL OACI': 'E145_DEP'}, axis=1)
    E145_ARR = E145_DIR.get_group('ARR')
    E145_ARR = pd.DataFrame(E145_ARR, columns=['SEL OACI'])
    E145_ARR = E145_ARR.rename({'SEL OACI': 'E145_ARR'}, axis=1)
    # E145_UNK = E145_DIR.get_group('UNK')
    # E145_UNK = pd.DataFrame(E145_UNK, columns=['SEL OACI'])
    # E145_UNK = E145_UNK.rename({'SEL OACI': 'E145_UNK'}, axis=1)
    BE20 = (df.loc[df['FIS Aircraft type'] == 'BE20'])
    BE20_DIR = BE20.groupby('Direction')
    BE20_DEP = BE20_DIR.get_group('DEP')
    BE20_DEP = pd.DataFrame(BE20_DEP, columns=['SEL OACI'])
    BE20_DEP = BE20_DEP.rename({'SEL OACI': 'BE20_DEP'}, axis=1)
    BE20_ARR = BE20_DIR.get_group('ARR')
    BE20_ARR = pd.DataFrame(BE20_ARR, columns=['SEL OACI'])
    BE20_ARR = BE20_ARR.rename({'SEL OACI': 'BE20_ARR'}, axis=1)
    # BE20_UNK = BE20_DIR.get_group('UNK')
    # BE20_UNK = pd.DataFrame(BE20_UNK, columns=['SEL OACI'])
    # BE20_UNK = BE20_UNK.rename({'SEL OACI': 'BE20_UNK'}, axis=1)
    C208 = (df.loc[df['FIS Aircraft type'] == 'C208'])
    C208_DIR = C208.groupby('Direction')
    # C208_DEP = C208_DIR.get_group('DEP')
    # C208_DEP = pd.DataFrame(C208_DEP, columns=['SEL OACI'])
    # C208_DEP = C208_DEP.rename({'SEL OACI': 'C208_DEP'}, axis=1)
    # C208_ARR = C208_DIR.get_group('ARR')
    # C208_ARR = pd.DataFrame(C208_ARR, columns=['SEL OACI'])
    # C208_ARR = C208_ARR.rename({'SEL OACI': 'C208_ARR'}, axis=1)
    # C208_UNK = C208_DIR.get_group('UNK')
    # C208_UNK = pd.DataFrame(C208_UNK, columns=['SEL OACI'])
    # C208_UNK = C208_UNK.rename({'SEL OACI': 'C208_UNK'}, axis=1)
    A30B = (df.loc[df['FIS Aircraft type'] == 'A30B'])  # Aca voy---------
    A30B_DIR = A30B.groupby('Direction')
    # A30B_DEP = A30B_DIR.get_group('DEP')
    # A30B_DEP = pd.DataFrame(A30B_DEP, columns=['SEL OACI'])
    # A30B_DEP = A30B_DEP.rename({'SEL OACI': 'A30B_DEP'}, axis=1)
    # A30B_ARR = A30B_DIR.get_group('ARR')
    # A30B_ARR = pd.DataFrame(A30B_ARR, columns=['SEL OACI'])
    # A30B_ARR = A30B_ARR.rename({'SEL OACI': 'A30B_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    B752 = (df.loc[df['FIS Aircraft type'] == 'B752'])
    B752_DIR = B752.groupby('Direction')
    # B752_DEP = B752_DIR.get_group('DEP')
    # B752_DEP = pd.DataFrame(B752_DEP, columns=['SEL OACI'])
    # B752_DEP = B752_DEP.rename({'SEL OACI': 'B752_DEP'}, axis=1)
    # B752_ARR = B752_DIR.get_group('ARR')
    # B752_ARR = pd.DataFrame(B752_ARR, columns=['SEL OACI'])
    # B752_ARR = B752_ARR.rename({'SEL OACI': 'B752_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    LJ45 = (df.loc[df['FIS Aircraft type'] == 'LJ45'])
    LJ45_DIR = LJ45.groupby('Direction')
    LJ45_DEP = LJ45_DIR.get_group('DEP')
    LJ45_DEP = pd.DataFrame(LJ45_DEP, columns=['SEL OACI'])
    LJ45_DEP = LJ45_DEP.rename({'SEL OACI': 'LJ45_DEP'}, axis=1)
    # LJ45_ARR = LJ45_DIR.get_group('ARR')
    # LJ45_ARR = pd.DataFrame(LJ45_ARR, columns=['SEL OACI'])
    # LJ45_ARR = LJ45_ARR.rename({'SEL OACI': 'LJ45_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    JS32 = (df.loc[df['FIS Aircraft type'] == 'JS32'])
    JS32_DIR = AT45.groupby('Direction')
    JS32_DEP = JS32_DIR.get_group('DEP')
    JS32_DEP = pd.DataFrame(JS32_DEP, columns=['SEL OACI'])
    JS32_DEP = AT45_DEP.rename({'SEL OACI': 'JS32_DEP'}, axis=1)
    JS32_ARR = JS32_DIR.get_group('ARR')
    JS32_ARR = pd.DataFrame(JS32_ARR, columns=['SEL OACI'])
    JS32_ARR = JS32_ARR.rename({'SEL OACI': 'JS32_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    B77L = (df.loc[df['FIS Aircraft type'] == 'B77L'])
    B77L_DIR = B77L.groupby('Direction')
    B77L_DEP = B77L_DIR.get_group('DEP')
    B77L_DEP = pd.DataFrame(B77L_DEP, columns=['SEL OACI'])
    B77L_DEP = B77L_DEP.rename({'SEL OACI': 'B77L_DEP'}, axis=1)
    B77L_ARR = B77L_DIR.get_group('ARR')
    B77L_ARR = pd.DataFrame(B77L_ARR, columns=['SEL OACI'])
    B77L_ARR = B77L_ARR.rename({'SEL OACI': 'B77L_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    B762 = (df.loc[df['FIS Aircraft type'] == 'B762'])
    B762_DIR = B762.groupby('Direction')
    # B762_DEP = B762_DIR.get_group('DEP')
    # B762_DEP = pd.DataFrame(B762_DEP, columns=['SEL OACI'])
    # B762_DEP = B762_DEP.rename({'SEL OACI': 'B762_DEP'}, axis=1)
    # B762_ARR = B762_DIR.get_group('ARR')
    # B762_ARR = pd.DataFrame(B762_ARR, columns=['SEL OACI'])
    # B762_ARR = B762_ARR.rename({'SEL OACI': 'B762_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    B733 = (df.loc[df['FIS Aircraft type'] == 'B733'])
    B733_DIR = B733.groupby('Direction')
    B733_DEP = B733_DIR.get_group('DEP')
    B733_DEP = pd.DataFrame(B733_DEP, columns=['SEL OACI'])
    B733_DEP = B733_DEP.rename({'SEL OACI': 'B733_DEP'}, axis=1)
    B733_ARR = B733_DIR.get_group('ARR')
    B733_ARR = pd.DataFrame(B733_ARR, columns=['SEL OACI'])
    B733_ARR = B733_ARR.rename({'SEL OACI': 'B733_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    A343 = (df.loc[df['FIS Aircraft type'] == 'A343'])
    A343_DIR = A343.groupby('Direction')
    # A343_DEP = A343_DIR.get_group('DEP')
    # A343_DEP = pd.DataFrame(A343_DEP, columns=['SEL OACI'])
    # A343_DEP = A343_DEP.rename({'SEL OACI': 'A343_DEP'}, axis=1)
    # A343_ARR = A343_DIR.get_group('ARR')
    # A343_ARR = pd.DataFrame(A343_ARR, columns=['SEL OACI'])
    # A343_ARR = A343_ARR.rename({'SEL OACI': 'A343_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    B350 = (df.loc[df['FIS Aircraft type'] == 'B350'])
    B350_DIR = B350.groupby('Direction')
    B350_DEP = B350_DIR.get_group('DEP')
    B350_DEP = pd.DataFrame(B350_DEP, columns=['SEL OACI'])
    B350_DEP = B350_DEP.rename({'SEL OACI': 'B350_DEP'}, axis=1)
    B350_ARR = B350_DIR.get_group('ARR')
    B350_ARR = pd.DataFrame(B350_ARR, columns=['SEL OACI'])
    B350_ARR = B350_ARR.rename({'SEL OACI': 'B350_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    B748 = (df.loc[df['FIS Aircraft type'] == 'B748'])
    B748_DIR = B748.groupby('Direction')
    # B748_DEP = B748_DIR.get_group('DEP')
    # B748_DEP = pd.DataFrame(B748_DEP, columns=['SEL OACI'])
    # B748_DEP = B748_DEP.rename({'SEL OACI': 'B748_DEP'}, axis=1)
    # B748_ARR = B748_DIR.get_group('ARR')
    # B748_ARR = pd.DataFrame(B748_ARR, columns=['SEL OACI'])
    # B748_ARR = B748_ARR.rename({'SEL OACI': 'B748_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    BE9L = (df.loc[df['FIS Aircraft type'] == 'BE9L'])
    BE9L_DIR = BE9L.groupby('Direction')
    BE9L_DEP = BE9L_DIR.get_group('DEP')
    BE9L_DEP = pd.DataFrame(BE9L_DEP, columns=['SEL OACI'])
    BE9L_DEP = BE9L_DEP.rename({'SEL OACI': 'BE9L_DEP'}, axis=1)
    BE9L_ARR = BE9L_DIR.get_group('ARR')
    BE9L_ARR = pd.DataFrame(BE9L_ARR, columns=['SEL OACI'])
    BE9L_ARR = BE9L_ARR.rename({'SEL OACI': 'BE9L_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    A330 = (df.loc[df['FIS Aircraft type'] == 'A330'])
    A330_DIR = A330.groupby('Direction')
    # A330_DEP = A330_DIR.get_group('DEP')
    # A330_DEP = pd.DataFrame(A330_DEP, columns=['SEL OACI'])
    # A330_DEP = A330_DEP.rename({'SEL OACI': 'A330_DEP'}, axis=1)
    # A330_ARR = A330_DIR.get_group('ARR')
    # A330_ARR = pd.DataFrame(A330_ARR, columns=['SEL OACI'])
    # A330_ARR = A330_ARR.rename({'SEL OACI': 'A330_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    E332 = (df.loc[df['FIS Aircraft type'] == 'E332'])
    E332_DIR = E332.groupby('Direction')
    # E332_DEP = E332_DIR.get_group('DEP')
    # E332_DEP = pd.DataFrame(E332_DEP, columns=['SEL OACI'])
    # E332_DEP = E332_DEP.rename({'SEL OACI': 'E332_DEP'}, axis=1)
    # E332_ARR = E332_DIR.get_group('ARR')
    # E332_ARR = pd.DataFrame(E332_ARR, columns=['SEL OACI'])
    # E332_ARR = E332_ARR.rename({'SEL OACI': 'E332_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    BE40 = (df.loc[df['FIS Aircraft type'] == 'BE40'])
    BE40_DIR = BE40.groupby('Direction')
    # BE40_DEP = BE40_DIR.get_group('DEP')
    # BE40_DEP = pd.DataFrame(BE40_DEP, columns=['SEL OACI'])
    # BE40_DEP = BE40_DEP.rename({'SEL OACI': 'BE40_DEP'}, axis=1)
    # BE40_ARR = BE40_DIR.get_group('ARR')
    # BE40_ARR = pd.DataFrame(BE40_ARR, columns=['SEL OACI'])
    # BE40_ARR = BE40_ARR.rename({'SEL OACI': 'BE40_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    B39M = (df.loc[df['FIS Aircraft type'] == 'B39M'])
    B39M_DIR = B39M.groupby('Direction')
    # B39M_DEP = B39M_DIR.get_group('DEP')
    # B39M_DEP = pd.DataFrame(B39M_DEP, columns=['SEL OACI'])
    # B39M_DEP = B39M_DEP.rename({'SEL OACI': 'B39M_DEP'}, axis=1)
    # B39M_ARR = B39M_DIR.get_group('ARR')
    # B39M_ARR = pd.DataFrame(B39M_ARR, columns=['SEL OACI'])
    # B39M_ARR = B39M_ARR.rename({'SEL OACI': 'B39M_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    B777 = (df.loc[df['FIS Aircraft type'] == 'B777'])
    B777_DIR = B777.groupby('Direction')
    # B777_DEP = B777_DIR.get_group('DEP')
    # B777_DEP = pd.DataFrame(B777_DEP, columns=['SEL OACI'])
    # B777_DEP = B777_DEP.rename({'SEL OACI': 'B777_DEP'}, axis=1)
    # B777_ARR = B777_DIR.get_group('ARR')
    # B777_ARR = pd.DataFrame(B777_ARR, columns=['SEL OACI'])
    # B777_ARR = B777_ARR.rename({'SEL OACI': 'B777_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    H25B = (df.loc[df['FIS Aircraft type'] == 'H25B'])
    H25B_DIR = H25B.groupby('Direction')
    # H25B__DEP = H25B_DIR.get_group('DEP')
    # H25B__DEP = pd.DataFrame(H25B__DEP, columns=['SEL OACI'])
    # H25B__DEP = H25B__DEP.rename({'SEL OACI': 'H25B__DEP'}, axis=1)
    # H25B__ARR = H25B_DIR.get_group('ARR')
    # H25B__ARR = pd.DataFrame(H25B__ARR, columns=['SEL OACI'])
    # H25B__ARR = H25B__ARR.rename({'SEL OACI': 'H25B__ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    CL60 = (df.loc[df['FIS Aircraft type'] == 'CL60'])
    CL60_DIR = CL60.groupby('Direction')
    # CL60_DEP = CL60_DIR.get_group('DEP')
    # CL60_DEP = pd.DataFrame(CL60_DEP, columns=['SEL OACI'])
    # CL60_DEP = CL60_DEP.rename({'SEL OACI': 'CL60_DEP'}, axis=1)
    # CL60_ARR = CL60_DIR.get_group('ARR')
    # CL60_ARR = pd.DataFrame(CL60_ARR, columns=['SEL OACI'])
    # CL60_ARR = CL60_ARR.rename({'SEL OACI': 'CL60_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    SF50 = (df.loc[df['FIS Aircraft type'] == 'SF50'])
    SF50_DIR = SF50.groupby('Direction')
    # SF50_DEP = SF50_DIR.get_group('DEP')
    # SF50_DEP = pd.DataFrame(SF50_DEP, columns=['SEL OACI'])
    # SF50_DEP = SF50_DEP.rename({'SEL OACI': 'SF50_DEP'}, axis=1)
    # SF50_ARR = SF50_DIR.get_group('ARR')
    # SF50_ARR = pd.DataFrame(SF50_ARR, columns=['SEL OACI'])
    # SF50_ARR = SF50_ARR.rename({'SEL OACI': 'SF50_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    AC90 = (df.loc[df['FIS Aircraft type'] == 'AC90'])
    AC90_DIR = AC90.groupby('Direction')
    # AC90_DEP = AC90_DIR.get_group('DEP')
    # AC90_DEP = pd.DataFrame(AC90_DEP, columns=['SEL OACI'])
    # AC90_DEP = AC90_DEP.rename({'SEL OACI': 'AC90_DEP'}, axis=1)
    # AC90_ARR = AC90_DIR.get_group('ARR')
    # AC90_ARR = pd.DataFrame(AC90_ARR, columns=['SEL OACI'])
    # AC90_ARR = AC90_ARR.rename({'SEL OACI': 'AC90_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    GALX = (df.loc[df['FIS Aircraft type'] == 'GALX'])
    GALX_DIR = GALX.groupby('Direction')
    # GALX_DEP = GALX_DIR.get_group('DEP')
    # GALX_DEP = pd.DataFrame(GALX_DEP, columns=['SEL OACI'])
    # GALX_DEP = GALX_DEP.rename({'SEL OACI': 'GALX_DEP'}, axis=1)
    # GALX_ARR = GALX_DIR.get_group('ARR')
    # GALX_ARR = pd.DataFrame(GALX_ARR, columns=['SEL OACI'])
    # GALX_ARR = GALX_ARR.rename({'SEL OACI': 'GALX_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)
    GLF6 = (df.loc[df['FIS Aircraft type'] == 'GLF6'])
    GLF6_DIR = GLF6.groupby('Direction')
    # GLF6_DEP = GLF6_DIR.get_group('DEP')
    # GLF6_DEP = pd.DataFrame(GLF6_DEP, columns=['SEL OACI'])
    # GLF6_DEP = GLF6_DEP.rename({'SEL OACI': 'GLF6_DEP'}, axis=1)
    # GLF6_ARR = GLF6_DIR.get_group('ARR')
    # GLF6_ARR = pd.DataFrame(GLF6_ARR, columns=['SEL OACI'])
    # GLF6_ARR = GLF6_ARR.rename({'SEL OACI': 'GLF6_ARR'}, axis=1)
    # AT45_UNK = AT45_DIR.get_group('UNK')
    # AT45_UNK = pd.DataFrame(AT45_UNK, columns=['SEL OACI'])
    # AT45_UNK = AT45_UNK.rename({'SEL OACI': 'AT45_UNK'}, axis=1)

l=0
for n in df:
    index_list_all[l]=(df.loc[df['FIS Aircraft type'] == index_list_all[l]])#Organiza todos los eventos por tipo de aeronave index_list_all[0].iloc[:,0]
    print(l)
    l=l+1

"""Sepracion de niveles diurnos para Graficos (Histogramas y Densidad de Kernel)"""
"""Dataframes para Histograma de LAeq, D/N"""
dates=pd.date_range(start='2022-12-01', end='2022-12-31', freq="D")#Rango de fechas del mes a procesar
o_date = dates.date
dates=o_date

F_1=df_ld.groupby('NMT')#Agrupa los datos en tipo NoneType
F1=F_1.get_group('F001')#Agrupa los datos por estacion
F1=pd.DataFrame(F1, columns=['LEVEL_LD'])#Agrupa niveles diurnos en tipo DataFrame
#F1=F1['LEVEL_LD']#Deja el index de la estacion y los valores diurnos en tipo series o vector
F1=F1.rename({'LEVEL_LD': 'F1'}, axis=1)

F_2=df_ld.groupby('NMT')
F2=F_2.get_group('F002')
F2=pd.DataFrame(F2, columns=['LEVEL_LD'])
F2=F2.rename({'LEVEL_LD': 'F2'}, axis=1)

F_3=df_ld.groupby('NMT')
F3=F_3.get_group('F003')
F3=pd.DataFrame(F3, columns=['LEVEL_LD'])
F3=F3.rename({'LEVEL_LD': 'F3'}, axis=1)

F_4=df_ld.groupby('NMT')
F4=F_4.get_group('F004')
F4=pd.DataFrame(F4, columns=['LEVEL_LD'])
F4=F4.rename({'LEVEL_LD': 'F4'}, axis=1)

F_5=df_ld.groupby('NMT')
F5=F_5.get_group('F005')
F5=pd.DataFrame(F5, columns=['LEVEL_LD'])
F5=F5.rename({'LEVEL_LD': 'F5'}, axis=1)

F_7=df_ld.groupby('NMT')
F7=F_7.get_group('F007')
F7=pd.DataFrame(F7, columns=['LEVEL_LD'])
F7=F7.rename({'LEVEL_LD': 'F7'}, axis=1)

F_8=df_ld.groupby('NMT')
F8=F_8.get_group('F008')
F8=pd.DataFrame(F8, columns=['LEVEL_LD'])
F8=F8.rename({'LEVEL_LD': 'F8'}, axis=1)

F_10=df_ld.groupby('NMT')
F10=F_10.get_group('F010')
F10=pd.DataFrame(F10, columns=['LEVEL_LD'])
F10=F10.rename({'LEVEL_LD': 'F10'}, axis=1)

F_11=df_ld.groupby('NMT')
F11=F_11.get_group('F011')
F11=pd.DataFrame(F11, columns=['LEVEL_LD'])
F11=F11.rename({'LEVEL_LD': 'F11'}, axis=1)

F_13=df_ld.groupby('NMT')
F13=F_13.get_group('F013')
F13=pd.DataFrame(F13, columns=['LEVEL_LD'])
F13=F13.rename({'LEVEL_LD': 'F13'}, axis=1)

F_15=df_ld.groupby('NMT')
F15=F_15.get_group('F015')
F15=pd.DataFrame(F15, columns=['LEVEL_LD'])
F15=F15.rename({'LEVEL_LD': 'F15'}, axis=1)

F_17=df_ld.groupby('NMT')
F17=F_17.get_group('F017')
F17=pd.DataFrame(F17, columns=['LEVEL_LD'])
F17=F17.rename({'LEVEL_LD': 'F17'}, axis=1)

F_18=df_ld.groupby('NMT')
F18=F_18.get_group('F018')
F18=pd.DataFrame(F18, columns=['LEVEL_LD'])
F18=F18.rename({'LEVEL_LD': 'F18'}, axis=1)

F_19=df_ld.groupby('NMT')
F19=F_19.get_group('F019')
F19=pd.DataFrame(F19, columns=['LEVEL_LD'])
F19=F19.rename({'LEVEL_LD': 'F19'}, axis=1)

F_20=df_ld.groupby('NMT')
F20=F_20.get_group('F020')
F20=pd.DataFrame(F20, columns=['LEVEL_LD'])
F20=F20.rename({'LEVEL_LD': 'F20'}, axis=1)

F_21=df_ld.groupby('NMT')
F21=F_21.get_group('F021')
F21=pd.DataFrame(F21, columns=['LEVEL_LD'])
F21=F21.rename({'LEVEL_LD': 'F21'}, axis=1)

F_23=df_ld.groupby('NMT')
F23=F_23.get_group('F023')
F23=pd.DataFrame(F23, columns=['LEVEL_LD'])
F23=F23.rename({'LEVEL_LD': 'F23'}, axis=1)

F_24=df_ld.groupby('NMT')
F24=F_24.get_group('F024')
F24=pd.DataFrame(F24, columns=['LEVEL_LD'])
F24=F24.rename({'LEVEL_LD': 'F24'}, axis=1)

F_25=df_ld.groupby('NMT')
F25=F_25.get_group('F025')
F25=pd.DataFrame(F25, columns=['LEVEL_LD'])
F25=F25.rename({'LEVEL_LD': 'F25'}, axis=1)

F_27=df_ld.groupby('NMT')
F27=F_27.get_group('F027')
F27=pd.DataFrame(F27, columns=['LEVEL_LD'])
F27=F27.rename({'LEVEL_LD': 'F27'}, axis=1)

F_28=df_ld.groupby('NMT')
F28=F_28.get_group('F028')
F28=pd.DataFrame(F28, columns=['LEVEL_LD'])
F28=F28.rename({'LEVEL_LD': 'F28'}, axis=1)

F_29=df_ld.groupby('NMT')
F29=F_29.get_group('F029')
F29=pd.DataFrame(F29, columns=['LEVEL_LD'])
F29=F29.rename({'LEVEL_LD': 'F29'}, axis=1)

F_30=df_ld.groupby('NMT')
F30=F_30.get_group('F030')
F30=pd.DataFrame(F30, columns=['LEVEL_LD'])
F30=F30.rename({'LEVEL_LD': 'F30'}, axis=1)

F_32=df_ld.groupby('NMT')
F32=F_32.get_group('F032')
F32=pd.DataFrame(F32, columns=['LEVEL_LD'])
F32=F32.rename({'LEVEL_LD': 'F32'}, axis=1)

F_33=df_ld.groupby('NMT')
F33=F_33.get_group('F033')
F33=pd.DataFrame(F33, columns=['LEVEL_LD'])
F33=F33.rename({'LEVEL_LD': 'F33'}, axis=1)

F_34=df_ld.groupby('NMT')
F34=F_34.get_group('F034')
F34=pd.DataFrame(F34, columns=['LEVEL_LD'])
F34=F34.rename({'LEVEL_LD': 'F34'}, axis=1)

"""Estándar Máximo Res 0627"""

lim_est_0627_ld = pd.DataFrame({'F1' :  np.repeat((75), len(dates)),
                   'F2': np.repeat((65), len(dates)),
                   'F3': np.repeat((65), len(dates)),
                   'F4': np.repeat((55), len(dates)),
                   'F5': np.repeat((55), len(dates)),
                   'F7': np.repeat((70), len(dates)),
                   'F8': np.repeat((70), len(dates)),
                   'F10': np.repeat((65), len(dates)),
                   'F11': np.repeat((55), len(dates)),
                   'F13': np.repeat((75), len(dates)),
                   'F15': np.repeat((75), len(dates)),
                   'F17': np.repeat((55), len(dates)),
                   'F18': np.repeat((65), len(dates)),
                   'F19': np.repeat((75), len(dates)),
                   'F20': np.repeat((75), len(dates)),
                   'F21': np.repeat((70), len(dates)),
                   'F23': np.repeat((65), len(dates)),
                   'F24': np.repeat((65), len(dates)),
                   'F25': np.repeat((75), len(dates)),
                   'F27': np.repeat((75), len(dates)),
                   'F28': np.repeat((75), len(dates)),
                   'F29': np.repeat((55), len(dates)),
                   'F30': np.repeat((70), len(dates)),
                   'F32': np.repeat((75), len(dates)),
                   'F33': np.repeat((65), len(dates)),
                   'F34': np.repeat((65), len(dates))},
                   index=dates)

"""GRAFICO DE NIVEL MAXIMO DE APORTE DIARIO POR MODELO DE AERONAVE Y TIPO DE OPERACION"""

top_niv_air_perday_max=top_niv_air_perday_max.set_index(['Date'])
top_niv_air_perday_max=top_niv_air_perday_max.reindex(dates)
top_niv_air_perday_max=top_niv_air_perday_max.reset_index()
top_niv_air_perday_max=top_niv_air_perday_max.set_index(['Date','NMT','FIS Aircraft type','Direction'])
top_niv_air_perday_max=top_niv_air_perday_max.round(1)
fig_eng = plt.figure()
graph_top_max=top_niv_air_perday_max.plot.barh(y="LEVEL_Aircraft_type", use_index=True,grid=True)
graph_top_max.bar_label(graph_top_max.containers[0], label_type='edge',padding=0.5,fontsize = 10)
graph_top_max.set_title('Niveles máximos de contribución acústica diurnos - (Diciembre-2022)', fontsize=20, fontweight='bold')
graph_top_max.set_xlabel('LAeqMax, Day (dBA)', fontsize=17)
graph_top_max.set_ylabel('Días',fontsize = 17)
min_value = top_niv_air_perday_max['LEVEL_Aircraft_type'].min()
min_value_top = min_value.min()
max_value = top_niv_air_perday_max['LEVEL_Aircraft_type'].max()
max_value_top = max_value.max()
plt.xlim(min_value_top-1,max_value_top+1)
# plt.tight_layout()
plt.show()
plt.savefig('Niv_Max_D.png')

"""DATAFRAMES CON NIVELES LEQ DE EVENTOS PARA CADA ESTACION"""

f1_leq_event=f1_events.groupby('NMT')
f1_leq_event=f1_leq_event.get_group('F001')
f1_leq_event=pd.DataFrame(f1_leq_event, columns=['LEQ OACI'])
f1_leq_event=f1_leq_event.rename({'LEQ OACI': 'F1_LEQ'}, axis=1)

f2_leq_event=f2_events.groupby('NMT')
f2_leq_event=f2_leq_event.get_group('F002')
f2_leq_event=pd.DataFrame(f2_leq_event, columns=['LEQ OACI'])
f2_leq_event=f2_leq_event.rename({'LEQ OACI': 'F2_LEQ'}, axis=1)

f3_leq_event=f3_events.groupby('NMT')
f3_leq_event=f3_leq_event.get_group('F003')
f3_leq_event=pd.DataFrame(f3_leq_event, columns=['LEQ OACI'])
f3_leq_event=f3_leq_event.rename({'LEQ OACI': 'F3_LEQ'}, axis=1)

f4_leq_event=f4_events.groupby('NMT')
f4_leq_event=f4_leq_event.get_group('F004')
f4_leq_event=pd.DataFrame(f4_leq_event, columns=['LEQ OACI'])
f4_leq_event=f4_leq_event.rename({'LEQ OACI': 'F4_LEQ'}, axis=1)

f5_leq_event=f5_events.groupby('NMT')
f5_leq_event=f5_leq_event.get_group('F005')
f5_leq_event=pd.DataFrame(f5_leq_event, columns=['LEQ OACI'])
f5_leq_event=f5_leq_event.rename({'LEQ OACI': 'F5_LEQ'}, axis=1)

f7_leq_event=f7_events.groupby('NMT')
f7_leq_event=f7_leq_event.get_group('F007')
f7_leq_event=pd.DataFrame(f7_leq_event, columns=['LEQ OACI'])
f7_leq_event=f7_leq_event.rename({'LEQ OACI': 'F7_LEQ'}, axis=1)

f8_leq_event=f8_events.groupby('NMT')
f8_leq_event=f8_leq_event.get_group('F008')
f8_leq_event=pd.DataFrame(f8_leq_event, columns=['LEQ OACI'])
f8_leq_event=f8_leq_event.rename({'LEQ OACI': 'F8_LEQ'}, axis=1)

f10_leq_event=f10_events.groupby('NMT')
f10_leq_event=f10_leq_event.get_group('F010')
f10_leq_event=pd.DataFrame(f10_leq_event, columns=['LEQ OACI'])
f10_leq_event=f10_leq_event.rename({'LEQ OACI': 'F10_LEQ'}, axis=1)

f11_leq_event=f11_events.groupby('NMT')
f11_leq_event=f11_leq_event.get_group('F011')
f11_leq_event=pd.DataFrame(f11_leq_event, columns=['LEQ OACI'])
f11_leq_event=f11_leq_event.rename({'LEQ OACI': 'F11_LEQ'}, axis=1)

f13_leq_event=f13_events.groupby('NMT')
f13_leq_event=f13_leq_event.get_group('F013')
f13_leq_event=pd.DataFrame(f13_leq_event, columns=['LEQ OACI'])
f13_leq_event=f13_leq_event.rename({'LEQ OACI': 'F13_LEQ'}, axis=1)

f15_leq_event=f15_events.groupby('NMT')
f15_leq_event=f15_leq_event.get_group('F015')
f15_leq_event=pd.DataFrame(f15_leq_event, columns=['LEQ OACI'])
f15_leq_event=f15_leq_event.rename({'LEQ OACI': 'F15_LEQ'}, axis=1)

f17_leq_event=f17_events.groupby('NMT')
f17_leq_event=f17_leq_event.get_group('F017')
f17_leq_event=pd.DataFrame(f17_leq_event, columns=['LEQ OACI'])
f17_leq_event=f17_leq_event.rename({'LEQ OACI': 'F17_LEQ'}, axis=1)

f18_leq_event=f18_events.groupby('NMT')
f18_leq_event=f18_leq_event.get_group('F018')
f18_leq_event=pd.DataFrame(f18_leq_event, columns=['LEQ OACI'])
f18_leq_event=f18_leq_event.rename({'LEQ OACI': 'F18_LEQ'}, axis=1)

f19_leq_event=f19_events.groupby('NMT')
f19_leq_event=f19_leq_event.get_group('F019')
f19_leq_event=pd.DataFrame(f19_leq_event, columns=['LEQ OACI'])
f19_leq_event=f19_leq_event.rename({'LEQ OACI': 'F19_LEQ'}, axis=1)

f20_leq_event=f20_events.groupby('NMT')
f20_leq_event=f20_leq_event.get_group('F020')
f20_leq_event=pd.DataFrame(f20_leq_event, columns=['LEQ OACI'])
f20_leq_event=f20_leq_event.rename({'LEQ OACI': 'F20_LEQ'}, axis=1)

f21_leq_event=f21_events.groupby('NMT')
f21_leq_event=f21_leq_event.get_group('F021')
f21_leq_event=pd.DataFrame(f21_leq_event, columns=['LEQ OACI'])
f21_leq_event=f21_leq_event.rename({'LEQ OACI': 'F21_LEQ'}, axis=1)

f23_leq_event=f23_events.groupby('NMT')
f23_leq_event=f23_leq_event.get_group('F023')
f23_leq_event=pd.DataFrame(f23_leq_event, columns=['LEQ OACI'])
f23_leq_event=f23_leq_event.rename({'LEQ OACI': 'F23_LEQ'}, axis=1)

f24_leq_event=f24_events.groupby('NMT')
f24_leq_event=f24_leq_event.get_group('F024')
f24_leq_event=pd.DataFrame(f24_leq_event, columns=['LEQ OACI'])
f24_leq_event=f24_leq_event.rename({'LEQ OACI': 'F24_LEQ'}, axis=1)

f25_leq_event=f25_events.groupby('NMT')
f25_leq_event=f25_leq_event.get_group('F025')
f25_leq_event=pd.DataFrame(f25_leq_event, columns=['LEQ OACI'])
f25_leq_event=f25_leq_event.rename({'LEQ OACI': 'F25_LEQ'}, axis=1)

f27_leq_event=f27_events.groupby('NMT')
f27_leq_event=f27_leq_event.get_group('F027')
f27_leq_event=pd.DataFrame(f27_leq_event, columns=['LEQ OACI'])
f27_leq_event=f27_leq_event.rename({'LEQ OACI': 'F27_LEQ'}, axis=1)

f28_leq_event=f28_events.groupby('NMT')
f28_leq_event=f28_leq_event.get_group('F028')
f28_leq_event=pd.DataFrame(f28_leq_event, columns=['LEQ OACI'])
f28_leq_event=f28_leq_event.rename({'LEQ OACI': 'F28_LEQ'}, axis=1)

f29_leq_event=f29_events.groupby('NMT')
f29_leq_event=f29_leq_event.get_group('F029')
f29_leq_event=pd.DataFrame(f29_leq_event, columns=['LEQ OACI'])
f29_leq_event=f29_leq_event.rename({'LEQ OACI': 'F29_LEQ'}, axis=1)

f30_leq_event=f30_events.groupby('NMT')
f30_leq_event=f30_leq_event.get_group('F030')
f30_leq_event=pd.DataFrame(f30_leq_event, columns=['LEQ OACI'])
f30_leq_event=f30_leq_event.rename({'LEQ OACI': 'F30_LEQ'}, axis=1)

f32_leq_event=f32_events.groupby('NMT')
f32_leq_event=f32_leq_event.get_group('F032')
f32_leq_event=pd.DataFrame(f32_leq_event, columns=['LEQ OACI'])
f32_leq_event=f32_leq_event.rename({'LEQ OACI': 'F32_LEQ'}, axis=1)

f33_leq_event=f33_events.groupby('NMT')
f33_leq_event=f33_leq_event.get_group('F033')
f33_leq_event=pd.DataFrame(f33_leq_event, columns=['LEQ OACI'])
f33_leq_event=f33_leq_event.rename({'LEQ OACI': 'F33_LEQ'}, axis=1)

f34_leq_event=f34_events.groupby('NMT')
f34_leq_event=f34_leq_event.get_group('F034')
f34_leq_event=pd.DataFrame(f34_leq_event, columns=['LEQ OACI'])
f34_leq_event=f34_leq_event.rename({'LEQ OACI': 'F34_LEQ'}, axis=1)

"""CONTEO DE NUEMRO DE EVENTOS POR ESTACION Y POR DIA"""

f1_LEN=f1_events.groupby(["Date"])['SEL'].count() #Esta linea cuenta el numero de eventos agrupados en f1_events
f1_LEN=f1_LEN.to_frame() #Esta linea convierte el conteo en un Dataframe
f1_LEN=f1_LEN.rename({'SEL': 'F1'}, axis=1) #Esta linea renombra la columna SEL por F1 o F#
f1_LEN = f1_LEN.reindex(dates) #Esta linea reindexa el conteo en las fechas establecidas en (dates)
f2_LEN=f2_events.groupby(["Date"])['SEL'].count()
f2_LEN=f2_LEN.to_frame()
f2_LEN=f2_LEN.rename({'SEL': 'F2'}, axis=1)
f2_LEN = f2_LEN.reindex(dates)
f3_LEN=f3_events.groupby(["Date"])['SEL'].count()
f3_LEN=f3_LEN.to_frame()
f3_LEN=f3_LEN.rename({'SEL': 'F3'}, axis=1)
f3_LEN = f3_LEN.reindex(dates)
f4_LEN=f4_events.groupby(["Date"])['SEL'].count()
f4_LEN=f4_LEN.to_frame()
f4_LEN=f4_LEN.rename({'SEL': 'F4'}, axis=1)
f4_LEN = f4_LEN.reindex(dates)
f5_LEN=f5_events.groupby(["Date"])['SEL'].count()
f5_LEN=f5_LEN.to_frame()
f5_LEN=f5_LEN.rename({'SEL': 'F5'}, axis=1)
f5_LEN = f5_LEN.reindex(dates)
f7_LEN=f7_events.groupby(["Date"])['SEL'].count()
f7_LEN=f7_LEN.to_frame()
f7_LEN=f7_LEN.rename({'SEL': 'F7'}, axis=1)
f7_LEN = f7_LEN.reindex(dates)
f8_LEN=f8_events.groupby(["Date"])['SEL'].count()
f8_LEN=f8_LEN.to_frame()
f8_LEN=f8_LEN.rename({'SEL': 'F8'}, axis=1)
f8_LEN = f8_LEN.reindex(dates)
f10_LEN=f10_events.groupby(["Date"])['SEL'].count()
f10_LEN=f10_LEN.to_frame()
f10_LEN=f10_LEN.rename({'SEL': 'F10'}, axis=1)
f10_LEN = f10_LEN.reindex(dates)
f11_LEN=f11_events.groupby(["Date"])['SEL'].count()
f11_LEN=f11_LEN.to_frame()
f11_LEN=f11_LEN.rename({'SEL': 'F11'}, axis=1)
f11_LEN = f11_LEN.reindex(dates)
f13_LEN=f13_events.groupby(["Date"])['SEL'].count()
f13_LEN=f13_LEN.to_frame()
f13_LEN=f13_LEN.rename({'SEL': 'F13'}, axis=1)
f13_LEN = f13_LEN.reindex(dates)
f15_LEN=f15_events.groupby(["Date"])['SEL'].count()
f15_LEN=f15_LEN.to_frame()
f15_LEN=f15_LEN.rename({'SEL': 'F15'}, axis=1)
f15_LEN = f15_LEN.reindex(dates)
f17_LEN=f17_events.groupby(["Date"])['SEL'].count()
f17_LEN=f17_LEN.to_frame()
f17_LEN=f17_LEN.rename({'SEL': 'F17'}, axis=1)
f17_LEN = f17_LEN.reindex(dates)
f18_LEN=f18_events.groupby(["Date"])['SEL'].count()
f18_LEN=f18_LEN.to_frame()
f18_LEN=f18_LEN.rename({'SEL': 'F18'}, axis=1)
f18_LEN = f18_LEN.reindex(dates)
f19_LEN=f19_events.groupby(["Date"])['SEL'].count()
f19_LEN=f19_LEN.to_frame()
f19_LEN=f19_LEN.rename({'SEL': 'F19'}, axis=1)
f19_LEN = f19_LEN.reindex(dates)
f20_LEN=f20_events.groupby(["Date"])['SEL'].count()
f20_LEN=f20_LEN.to_frame()
f20_LEN=f20_LEN.rename({'SEL': 'F20'}, axis=1)
f20_LEN = f20_LEN.reindex(dates)
f21_LEN=f21_events.groupby(["Date"])['SEL'].count()
f21_LEN=f21_LEN.to_frame()
f21_LEN=f21_LEN.rename({'SEL': 'F21'}, axis=1)
f21_LEN = f21_LEN.reindex(dates)
f23_LEN=f23_events.groupby(["Date"])['SEL'].count()
f23_LEN=f23_LEN.to_frame()
f23_LEN=f23_LEN.rename({'SEL': 'F23'}, axis=1)
f23_LEN = f23_LEN.reindex(dates)
f24_LEN=f24_events.groupby(["Date"])['SEL'].count()
f24_LEN=f24_LEN.to_frame()
f24_LEN=f24_LEN.rename({'SEL': 'F24'}, axis=1)
f24_LEN = f24_LEN.reindex(dates)
f25_LEN=f25_events.groupby(["Date"])['SEL'].count()
f25_LEN=f25_LEN.to_frame()
f25_LEN=f25_LEN.rename({'SEL': 'F25'}, axis=1)
f25_LEN = f25_LEN.reindex(dates)
f27_LEN=f27_events.groupby(["Date"])['SEL'].count()
f27_LEN=f27_LEN.to_frame()
f27_LEN=f27_LEN.rename({'SEL': 'F27'}, axis=1)
f27_LEN = f27_LEN.reindex(dates)
f28_LEN=f28_events.groupby(["Date"])['SEL'].count()
f28_LEN=f28_LEN.to_frame()
f28_LEN=f28_LEN.rename({'SEL': 'F28'}, axis=1)
f28_LEN = f28_LEN.reindex(dates)
f29_LEN=f29_events.groupby(["Date"])['SEL'].count()
f29_LEN=f29_LEN.to_frame()
f29_LEN=f29_LEN.rename({'SEL': 'F29'}, axis=1)
f29_LEN = f29_LEN.reindex(dates)
f30_LEN=f30_events.groupby(["Date"])['SEL'].count()
f30_LEN=f30_LEN.to_frame()
f30_LEN=f30_LEN.rename({'SEL': 'F30'}, axis=1)
f30_LEN = f30_LEN.reindex(dates)
f32_LEN=f32_events.groupby(["Date"])['SEL'].count()
f32_LEN=f32_LEN.to_frame()
f32_LEN=f32_LEN.rename({'SEL': 'F33'}, axis=1)
f32_LEN = f32_LEN.reindex(dates)
f33_LEN=f33_events.groupby(["Date"])['SEL'].count()
f33_LEN=f33_LEN.to_frame()
f33_LEN=f33_LEN.rename({'SEL': 'F33'}, axis=1)
f33_LEN = f33_LEN.reindex(dates)
f34_LEN=f34_events.groupby(["Date"])['SEL'].count()
f34_LEN=f34_LEN.to_frame()
f34_LEN=f34_LEN.rename({'SEL': 'F34'}, axis=1)
f34_LEN = f34_LEN.reindex(dates)

Count_Events = pd.concat([f1_LEN, f2_LEN, f3_LEN, f4_LEN, f5_LEN, f7_LEN, f8_LEN, f10_LEN, f11_LEN, f13_LEN, f15_LEN,
                          f17_LEN, f18_LEN,f19_LEN, f20_LEN, f21_LEN, f23_LEN, f24_LEN, f25_LEN, f27_LEN, f28_LEN,
                          f29_LEN, f30_LEN, f33_LEN, f32_LEN, f34_LEN],axis=1)

"""En esta seccion se completa cada Dataframe con el total de dias del mes de evaluacion"""

F1 = F1.reindex(dates).round(1)
F2 = F2.reindex(dates).round(1)
F3 = F3.reindex(dates).round(1)
F4 = F4.reindex(dates).round(1)
F5 = F5.reindex(dates).round(1)
F7 = F7.reindex(dates).round(1)
F8 = F8.reindex(dates).round(1)
F10 = F10.reindex(dates).round(1)
F11 = F11.reindex(dates).round(1)
F13 = F13.reindex(dates).round(1)
F15 = F15.reindex(dates).round(1)
F17 = F17.reindex(dates).round(1)
F18 = F18.reindex(dates).round(1)
F19 = F19.reindex(dates).round(1)
F20 = F20.reindex(dates).round(1)
F21 = F21.reindex(dates).round(1)
F23 = F23.reindex(dates).round(1)
F24 = F24.reindex(dates).round(1)
F25 = F25.reindex(dates).round(1)
F27 = F27.reindex(dates).round(1)
F28 = F28.reindex(dates).round(1)
F29 = F29.reindex(dates).round(1)
F30 = F30.reindex(dates).round(1)
F32 = F32.reindex(dates).round(1)
F33 = F33.reindex(dates).round(1)
F34 = F34.reindex(dates).round(1)

engativa=pd.DataFrame()
engativa=engativa.append(F1)

diurno=pd.concat([F1,F2,F3,F4,F5,F7,F8,F10,F11,F13,F15,F17,F18,F19,F20,F21,F23,F24,F25,F27,F28,F29,F30,F32,F33,F34],axis=1)
min_val = diurno.min().min()
diurno=diurno.round(1)

"""COMPARACIÓN DE NIVELES LINEA BASE 2019"""

lim_linea_base = pd.DataFrame({'F2':  np.repeat((66), len(dates)),
                   'F3': np.repeat((60), len(dates))}, index=dates)

emri2_3=pd.concat([F2,F3],axis=1)
emri2_3=emri2_3.round(1)

comp_linea_base=emri2_3.subtract(lim_linea_base)
bajo_lb=comp_linea_base.loc[comp_linea_base['F2'] <= 0]
prom_bajo_lb_f2=bajo_lb["F2"].mean()#Diferencia promedio de niveles por debajo de LB en F2
prom_bajo_lb_f3=bajo_lb["F3"].mean()#Diferencia promedio de niveles por debajo de LB en F3
sobre_lb=comp_linea_base.loc[comp_linea_base['F2'] >= 0]
prom_sobre_lb_f2=sobre_lb["F2"].mean()#Diferencia promedio de niveles sobre la LB en F2
prom_sobre_lb_f3=sobre_lb["F3"].mean()#Diferencia promedio de niveles sobre la LB en F3


"""COMPARACIÓN DE NIVELES RES 0627 (TABLA 1)"""

comp_res0627=diurno.subtract(lim_est_0627_ld)
bajo_res0627=comp_res0627.copy()
bajo_res0627.iloc[bajo_res0627 >= 0]=np.nan #Solo dias con niveles por debajo de Res0627
prom_bajo_res0627=bajo_res0627.mean()
sobre_res0627=comp_res0627.copy()
sobre_res0627.iloc[sobre_res0627 <= 0]=np.nan #Solo dias con niveles por debajo de Res0627
prom_sobre_res0627=sobre_res0627.mean()


av_comp_res0627 = comp_res0627.mean(axis=0)
av_comp_res0627=av_comp_res0627.round(1)
print(av_comp_res0627)

"""CREACIÓN DE TABLA DE FRECUENCIA"""
bins = np.arange(min_val,diurno.stack().max()+1,3.0)
freq_table=pd.concat([diurno.groupby(pd.cut(diurno.F1, bins=bins)).F1.count(),
                      diurno.groupby(pd.cut(diurno.F2, bins=bins)).F2.count(), 
                      diurno.groupby(pd.cut(diurno.F3, bins=bins)).F3.count(), 
                      diurno.groupby(pd.cut(diurno.F4, bins=bins)).F4.count(),
                      diurno.groupby(pd.cut(diurno.F5, bins=bins)).F5.count(),
                      diurno.groupby(pd.cut(diurno.F7, bins=bins)).F7.count(),
                      diurno.groupby(pd.cut(diurno.F8, bins=bins)).F8.count(),
                      diurno.groupby(pd.cut(diurno.F10, bins=bins)).F10.count(),
                      diurno.groupby(pd.cut(diurno.F11, bins=bins)).F11.count(),
                      diurno.groupby(pd.cut(diurno.F13, bins=bins)).F13.count(),
                      diurno.groupby(pd.cut(diurno.F15, bins=bins)).F15.count(),
                      diurno.groupby(pd.cut(diurno.F17, bins=bins)).F17.count(),
                      diurno.groupby(pd.cut(diurno.F18, bins=bins)).F18.count(),
                      diurno.groupby(pd.cut(diurno.F19, bins=bins)).F19.count(),
                      diurno.groupby(pd.cut(diurno.F20, bins=bins)).F20.count(),
                      diurno.groupby(pd.cut(diurno.F21, bins=bins)).F21.count(),
                      diurno.groupby(pd.cut(diurno.F23, bins=bins)).F23.count(),
                      diurno.groupby(pd.cut(diurno.F24, bins=bins)).F24.count(),
                      diurno.groupby(pd.cut(diurno.F25, bins=bins)).F25.count(),
                      diurno.groupby(pd.cut(diurno.F27, bins=bins)).F27.count(),
                      diurno.groupby(pd.cut(diurno.F28, bins=bins)).F28.count(),
                      diurno.groupby(pd.cut(diurno.F29, bins=bins)).F29.count(),
                      diurno.groupby(pd.cut(diurno.F30, bins=bins)).F30.count(),
                      diurno.groupby(pd.cut(diurno.F32, bins=bins)).F32.count(),
                      diurno.groupby(pd.cut(diurno.F33, bins=bins)).F33.count(),
                      diurno.groupby(pd.cut(diurno.F34, bins=bins)).F34.count()], axis = 1)

freq_table['%F1'] = 100 * freq_table['F1'] / freq_table['F1'].sum()
freq_table['%F2'] = 100 * freq_table['F2'] / freq_table['F2'].sum()
freq_table['%F3'] = 100 * freq_table['F3'] / freq_table['F3'].sum()
freq_table['%F4'] = 100 * freq_table['F4'] / freq_table['F4'].sum()
freq_table['%F5'] = 100 * freq_table['F5'] / freq_table['F5'].sum()
freq_table['%F7'] = 100 * freq_table['F7'] / freq_table['F7'].sum()
freq_table['%F8'] = 100 * freq_table['F8'] / freq_table['F8'].sum()
freq_table['%F10'] = 100 * freq_table['F10'] / freq_table['F10'].sum()
freq_table['%F11'] = 100 * freq_table['F11'] / freq_table['F11'].sum()
freq_table['%F13'] = 100 * freq_table['F13'] / freq_table['F13'].sum()
freq_table['%F15'] = 100 * freq_table['F15'] / freq_table['F15'].sum()
freq_table['%F17'] = 100 * freq_table['F17'] / freq_table['F17'].sum()
freq_table['%F18'] = 100 * freq_table['F18'] / freq_table['F18'].sum()
freq_table['%F19'] = 100 * freq_table['F19'] / freq_table['F19'].sum()
freq_table['%F20'] = 100 * freq_table['F20'] / freq_table['F20'].sum()
freq_table['%F21'] = 100 * freq_table['F21'] / freq_table['F21'].sum()
freq_table['%F23'] = 100 * freq_table['F23'] / freq_table['F23'].sum()
freq_table['%F24'] = 100 * freq_table['F24'] / freq_table['F24'].sum()
freq_table['%F25'] = 100 * freq_table['F25'] / freq_table['F25'].sum()
freq_table['%F27'] = 100 * freq_table['F27'] / freq_table['F27'].sum()
freq_table['%F28'] = 100 * freq_table['F28'] / freq_table['F28'].sum()
freq_table['%F29'] = 100 * freq_table['F29'] / freq_table['F29'].sum()
freq_table['%F30'] = 100 * freq_table['F30'] / freq_table['F30'].sum()
freq_table['%F32'] = 100 * freq_table['F32'] / freq_table['F32'].sum()
freq_table['%F33'] = 100 * freq_table['F33'] / freq_table['F33'].sum()
freq_table['%F34'] = 100 * freq_table['F34'] / freq_table['F34'].sum()

print(freq_table)

"""Graficas Jornadas - Engativa"""

engativa=pd.concat([F1,F2,F10,F18,F33],axis=1)

fig_eng = plt.figure()
ax_eng = engativa[['F1','F2','F10','F18','F33']].plot(kind='bar', use_index=True,grid=True)
# ax2_eng = ax_eng.twinx()#Crea un eje y secundario
ax_eng.plot(engativa[['F1','F2','F10','F18','F33']].values, linestyle='-', marker='o', linewidth=2.0)
ax_eng.set_title('Niveles diurnos de ruido aeronáutico - Localidad de Engativá - (Diciembre-2022)', fontsize = 20, fontweight='bold')
ax_eng.set_xlabel('Jornadas Diurnas',fontsize = 17)
ax_eng.set_ylabel('LAeq, Day (dBA)',fontsize = 17)
min_value = engativa.min()
min_value_eng = min_value.min()
max_value = engativa.max()
max_value_eng = max_value.max()
plt.ylim(min_value_eng-1,max_value_eng+1)
plt.tight_layout()
plt.show()
plt.savefig('ENG.png')

"""Graficas Jornadas - Fontibon"""

fontibon=pd.concat([F3,F7,F21,F23,F25,F27,F30],axis=1)

fig_fon = plt.figure()
ax_fon = fontibon[['F3','F7','F21','F23','F25','F27','F30']].plot(kind='bar', use_index=True,grid=True)
# ax2_fon = ax_fon.twinx()
ax_fon.plot(fontibon[['F3','F7','F21','F23','F25','F27','F30']].values, linestyle='-', marker='o', linewidth=2.0)
ax_fon.set_title('Niveles diurnos de ruido aeronáutico - Localidad de Fontibón -  (Diciembre-2022)', fontsize = 20, fontweight='bold')
ax_fon.set_xlabel('Jornadas Diurnas',fontsize = 17)
ax_fon.set_ylabel('LAeq, Day (dBA)',fontsize = 17)
min_value = fontibon.min()
min_value_fon = min_value.min()
max_value = fontibon.max()
max_value_fon = max_value.max()
plt.ylim(min_value_fon-1,max_value_fon+1)
plt.tight_layout()
plt.show()

"""Graficas Jornadas - Funza"""

funza=pd.concat([F4,F5,F11,F17,F24,F29],axis=1)

fig_fun = plt.figure()
ax_fun = funza[['F4','F5','F11','F17','F24','F29']].plot(kind='bar', use_index=True,grid=True)
# ax2_fon = ax_fon.twinx()
ax_fun.plot(funza[['F4','F5','F11','F17','F24','F29']].values, linestyle='-', marker='o', linewidth=2.0)
ax_fun.set_title('Niveles diurnos de ruido aeronáutico - Localidad de Funza -  (Diciembre-2022)', fontsize=20, fontweight='bold')
ax_fun.set_xlabel('Jornadas Diurnas', fontsize=17)
ax_fun.set_ylabel('LAeq, Day (dBA)', fontsize=17)
min_value = funza.min()
min_value_fun = min_value.min()
max_value = funza.max()
max_value_fun = max_value.max()
plt.ylim(min_value_fun-1,max_value_fun+3)
plt.tight_layout()
plt.show()

"""Graficas Jornadas - SKBO"""

skbo=pd.concat([F13,F15,F19,F20],axis=1)

fig_skbo = plt.figure()
ax_skbo = skbo[['F13','F15','F19','F20']].plot(kind='bar', use_index=True,grid=True)
# ax2_fon = ax_fon.twinx()
ax_skbo.plot(skbo[['F13','F15','F19','F20']].values, linestyle='-', marker='o', linewidth=2.0)
ax_skbo.set_title('Niveles diurnos de ruido aeronáutico - Aeropuerto Internacional El Dorado -  (Diciembre-2022)', fontsize=20, fontweight='bold')
ax_skbo.set_xlabel('Jornadas Diurnas', fontsize=17)
ax_skbo.set_ylabel('LAeq, Day (dBA)', fontsize=17)
min_value = skbo.min()
min_value_skbo = min_value.min()
max_value = skbo.max()
max_value_skbo = max_value.max()
plt.ylim(min_value_skbo-1,max_value_skbo+3)
plt.tight_layout()
plt.show()

"""Graficas Jornadas - Suba"""

suba=pd.concat([F34],axis=1)

fig_suba = plt.figure()
ax_suba = suba[['F34']].plot(kind='bar', use_index=True,grid=True)
# ax2_fon = ax_fon.twinx()
ax_suba.plot(suba[['F34']].values, linestyle='-', marker='o', linewidth=2.0)
ax_suba.set_title('Niveles diurnos de ruido aeronáutico - Localidad de Suba -  (Diciembre-2022)',fontsize=20, fontweight='bold')
ax_suba.set_xlabel('Jornadas Diurnas', fontsize=17)
ax_suba.set_ylabel('LAeq, Day (dBA)', fontsize=17)
min_value = suba.min()
min_value_suba = min_value.min()
max_value = suba.max()
max_value_suba = max_value.max()
plt.ylim(min_value_suba-1,max_value_suba+1)
plt.tight_layout()
plt.show()

"""ENGATIVA DIURNO CONO DE APROXIMACION"""

eng_ca_ld=pd.concat([F1,F2,F10,F18,F33],axis=1)
# Configuración de estilo
sns.set_style("whitegrid")
sns.set_palette("colorblind")
#Borrar datos NaN
eng_ca_ld = eng_ca_ld.dropna()
# Configurar los parámetros del histograma
bin_width = 1  # Ancho de los bins
alpha = 0.8  # Opacidad del histograma
density = True  # Normalizar el histograma

# Configurar los parámetros de las curvas de kernel
kernel_color = ['#FF5733','#8E44AD','#2980B9','#F1C40F','#D35400']  # Colores de las curvas de kernel
kernel_style = ['--', '-.', ':', '-.', ':']  # Estilos de las curvas de kernel
kernel_width = [2, 2, 2, 2, 2]  # Anchos de las curvas de kernel

# Crear el histograma y las curvas de kernel
fig, ax = plt.subplots(figsize=(12, 10))
for i in range(len(eng_ca_ld.columns)):
    sns.histplot(data=eng_ca_ld, x=eng_ca_ld.columns[i], color=kernel_color[i], multiple='dodge',
                 bins=int((max(eng_ca_ld.iloc[:, i]) - min(eng_ca_ld.iloc[:, i]))/ bin_width), alpha=alpha,
                 kde=False, label=eng_ca_ld.columns[i], linewidth=2, ax=ax, stat='density')
    kde = sns.kdeplot(data=eng_ca_ld, x=eng_ca_ld.columns[i], bw_method='silverman', color=kernel_color[i], multiple='stack',
                      linestyle=kernel_style[i], linewidth=kernel_width[i], fill=True, alpha=0.5, label=eng_ca_ld.columns[i],
                      ax=ax)

    # ax_eng_ca_ld = eng_ca_ld.sort_index(kde)
    # if len(ax_eng_ca_ld.right_ax.lines) > 0:
    #     kde = ax_eng_ca_ld.right_ax.lines[-1].get_data()
    #     print(kde[1])
    # else:
    #     print("No se encontró la curva de densidad de kernel en el gráfico.")
    # ax_eng_ca_ld.right_ax.set_ylim([0, kde[1].max() + 0.05])

# Configurar las leyendas y etiquetas del gráfico
plt.legend(title="Estaciones del SVCA", loc="upper right", labels=eng_ca_ld.columns, fontsize=12, frameon=True,
           facecolor='white', framealpha=0.8, markerscale=1)
plt.xlabel('LAeq, Day (dBA)', fontsize=17)
plt.ylabel('Frecuencia (Normalizada)', fontsize=17)
plt.title('HISTOGRAMA DE NIVELES DIURNOS DE RUIDO AERONÁUTICO CONO DE APROXIMACIÓN - ENGATIVÁ', fontsize=19, fontweight='bold')
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=10)

# Configurar el eje y secundario
ax2 = ax.twinx()
# ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
ax2.set_ylabel('Densidad de probabilidad de Kernel', fontsize=17)
ax2.grid(False)

# Configurar tamaño de ejes
min_value = eng_ca_ld.min()
max_value = eng_ca_ld.max()
min_value_eng_ca_ld = min_value.min()
max_value_eng_ca_ld = max_value.max()
plt.xlim(min_value_eng_ca_ld-2,max_value_eng_ca_ld+2)

# Mostrar el gráfico
# sns.despine()
plt.tight_layout()
plt.savefig('ENG_CA_LD.png')
plt.show()

"""ENGATIVA DIURNO COSTADO DE PISTA

eng_cp_ld=pd.concat([F8,F28],axis=1)
# Configuración de estilo
sns.set_style("whitegrid")
sns.set_palette("colorblind")
#Borrar datos NaN
eng_cp_ld = eng_cp_ld.dropna()
# Configurar los parámetros del histograma
bin_width = 1  # Ancho de los bins
alpha = 0.8  # Opacidad del histograma
density = True  # Normalizar el histograma

# Configurar los parámetros de las curvas de kernel
kernel_color = ['#FF5733','#8E44AD','#2980B9','#F1C40F','#D35400']  # Colores de las curvas de kernel
kernel_style = ['--', '-.', ':', '-.', ':']  # Estilos de las curvas de kernel
kernel_width = [2, 2, 2, 2, 2]  # Anchos de las curvas de kernel

# Crear el histograma y las curvas de kernel
fig, ax = plt.subplots(figsize=(12, 10))
for i in range(len(eng_cp_ld.columns)):
    sns.histplot(data=eng_ca_ld, x=eng_cp_ld.columns[i], color=kernel_color[i], multiple='dodge',
                 bins=int((max(eng_cp_ld.iloc[:, i]) - min(eng_cp_ld.iloc[:, i]))/ bin_width), alpha=alpha, kde=False,
                 label=eng_cp_ld.columns[i], linewidth=2, ax=ax, stat='density')
    kde = sns.kdeplot(data=eng_cp_ld, x=eng_cp_ld.columns[i], bw_method='silverman', color=kernel_color[i], multiple='stack',
                      linestyle=kernel_style[i], linewidth=kernel_width[i], fill=True, alpha=0.5, label=eng_cp_ld.columns[i],
                      ax=ax)

# Configurar las leyendas y etiquetas del gráfico
plt.legend(title="Estaciones del SVCA", loc="upper right", labels=eng_cp_ld.columns, fontsize=12, frameon=True,
           facecolor='white', framealpha=0.8, markerscale=1)
plt.xlabel('LAeq, Day (dBA)', fontsize=17)
plt.ylabel('Frecuencia (Normalizada)', fontsize=17)
plt.title('HISTOGRAMA DE NIVELES DIURNOS DE RUIDO AERONÁUTICO COSTADO DE PISTA - ENGATIVÁ', fontsize=19, fontweight='bold')
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=10)

# Configurar el eje y secundario
ax2 = ax.twinx()
# ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
ax2.set_ylabel('Densidad de probabilidad de Kernel', fontsize=17)
ax2.grid(False)

# Configurar tamaño de ejes
min_value = eng_cp_ld.min()
max_value = eng_cp_ld.max()
min_value_eng_cp_ld = min_value.min()
max_value_eng_cp_ld = max_value.max()
plt.xlim(min_value_eng_cp_ld-2,max_value_eng_cp_ld+2)

# Mostrar el gráfico
# sns.despine()
plt.tight_layout()
plt.savefig('ENG_CP_LD.png')
plt.show()"""


"""FONTIBON DIURNO CONO DE APROXIMACION"""

fon_ca_ld=pd.concat([F3,F21,F23,F25,F30],axis=1)
# Configuración de estilo
sns.set_style("whitegrid")
sns.set_palette("colorblind")
# Borrar datos NaN
fon_ca_ld = fon_ca_ld.dropna()
# Configurar los parámetros del histograma
bin_width = 1  # Ancho de los bins
alpha = 0.8  # Opacidad del histograma
density = True  # Normalizar el histograma

# Configurar los parámetros de las curvas de kernel
kernel_color = ['#FF5733', '#8E44AD', '#2980B9', '#F1C40F', '#D35400']  # Colores de las curvas de kernel
kernel_style = ['--', '-.', ':', '-.', ':']  # Estilos de las curvas de kernel
kernel_width = [2, 2, 2, 2, 2]  # Anchos de las curvas de kernel

# Crear el histograma y las curvas de kernel
fig, ax = plt.subplots(figsize=(12, 10))
for i in range(len(fon_ca_ld.columns)):
    sns.histplot(data=fon_ca_ld, x=fon_ca_ld.columns[i], color=kernel_color[i], multiple='dodge',
                 bins=int((max(fon_ca_ld.iloc[:, i]) - min(fon_ca_ld.iloc[:, i])) / bin_width), alpha=alpha, kde=False,
                 label=fon_ca_ld.columns[i], linewidth=2, ax=ax, stat='density')
    kde = sns.kdeplot(data=fon_ca_ld, x=fon_ca_ld.columns[i], bw_method='silverman', color=kernel_color[i], multiple='stack',
                      linestyle=kernel_style[i], linewidth=kernel_width[i], fill=True, alpha=0.5, label=fon_ca_ld.columns[i],
                      ax=ax)

# Configurar las leyendas y etiquetas del gráfico
plt.legend(title="Estaciones del SVCA", loc="upper right", labels=fon_ca_ld.columns, fontsize=12, frameon=True,
           facecolor='white', framealpha=0.8, markerscale=1)
plt.xlabel('LAeq, Day (dBA)', fontsize=17)
plt.ylabel('Frecuencia (Normalizada)', fontsize=17)
plt.title('HISTOGRAMA DE NIVELES DIURNOS DE RUIDO AERONÁUTICO CONO DE APROXIMACIÓN - FONTIBÓN', fontsize=19,
          fontweight='bold')
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=10)

# Configurar el eje y secundario
ax2 = ax.twinx()
# ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
ax2.set_ylabel('Densidad de probabilidad de Kernel', fontsize=17)
ax2.grid(False)

# Configurar tamaño de ejes
min_value = fon_ca_ld.min()
max_value = fon_ca_ld.max()
min_value_fon_ca_ld = min_value.min()
max_value_fon_ca_ld = max_value.max()
plt.xlim(min_value_fon_ca_ld - 2, max_value_fon_ca_ld + 2)

# Mostrar el gráfico
# sns.despine()
plt.tight_layout()
plt.savefig('FON_CA_LD.png')
plt.show()


"""FONTIBON DIURNO COSTADO DE PISTA"""

fon_cp_ld=pd.concat([F7,F27],axis=1)
# Configuración de estilo
sns.set_style("whitegrid")
sns.set_palette("colorblind")
# Borrar datos NaN
fon_cp_ld = fon_cp_ld.dropna()
# Configurar los parámetros del histograma
bin_width = 1  # Ancho de los bins
alpha = 0.8  # Opacidad del histograma
density = True  # Normalizar el histograma

# Configurar los parámetros de las curvas de kernel
kernel_color = ['#FF5733', '#8E44AD', '#2980B9', '#F1C40F', '#D35400']  # Colores de las curvas de kernel
kernel_style = ['--', '-.', ':', '-.', ':']  # Estilos de las curvas de kernel
kernel_width = [2, 2, 2, 2, 2]  # Anchos de las curvas de kernel

# Crear el histograma y las curvas de kernel
fig, ax = plt.subplots(figsize=(12, 10))
for i in range(len(fon_cp_ld.columns)):
    sns.histplot(data=fon_cp_ld, x=fon_cp_ld.columns[i], color=kernel_color[i], multiple='dodge',
                 bins=int((max(fon_cp_ld.iloc[:, i]) - min(fon_cp_ld.iloc[:, i])) / bin_width), alpha=alpha, kde=False,
                 label=fon_cp_ld.columns[i], linewidth=2, ax=ax, stat='density')
    kde = sns.kdeplot(data=fon_cp_ld, x=fon_cp_ld.columns[i], bw_method='silverman', color=kernel_color[i], multiple='stack',
                      linestyle=kernel_style[i], linewidth=kernel_width[i], fill=True, alpha=0.5, label=fon_cp_ld.columns[i],
                      ax=ax)

# Configurar las leyendas y etiquetas del gráfico
plt.legend(title="Estaciones del SVCA", loc="upper right", labels=fon_cp_ld.columns, fontsize=12, frameon=True,
           facecolor='white', framealpha=0.8, markerscale=1)
plt.xlabel('LAeq, Day (dBA)', fontsize=17)
plt.ylabel('Frecuencia (Normalizada)', fontsize=17)
plt.title('HISTOGRAMA DE NIVELES DIURNOS DE RUIDO AERONÁUTICO COSTADO DE PISTA - FONTIBÓN', fontsize=19,
          fontweight='bold')
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=10)

# Configurar el eje y secundario
ax2 = ax.twinx()
# ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
ax2.set_ylabel('Densidad de probabilidad de Kernel', fontsize=17)
ax2.grid(False)

# Configurar tamaño de ejes
min_value = fon_cp_ld.min()
max_value = fon_cp_ld.max()
min_value_fon_cp_ld = min_value.min()
max_value_fon_cp_ld = max_value.max()
plt.xlim(min_value_fon_cp_ld - 2, max_value_fon_cp_ld + 2)

# Mostrar el gráfico
# sns.despine()
plt.tight_layout()
plt.savefig('FON_CP_LD.png')
plt.show()

"""FUNZA DIURNO AREA RURAL"""

fun_rural_ld=pd.concat([F5,F11,F17],axis=1)
# Configuración de estilo
sns.set_style("whitegrid")
sns.set_palette("colorblind")
# Borrar datos NaN
fun_rural_ld = fun_rural_ld.dropna()
# Configurar los parámetros del histograma
bin_width = 1  # Ancho de los bins
alpha = 0.8  # Opacidad del histograma
density = True  # Normalizar el histograma

# Configurar los parámetros de las curvas de kernel
kernel_color = ['#FF5733', '#8E44AD', '#2980B9', '#F1C40F', '#D35400']  # Colores de las curvas de kernel
kernel_style = ['--', '-.', ':', '-.', ':']  # Estilos de las curvas de kernel
kernel_width = [2, 2, 2, 2, 2]  # Anchos de las curvas de kernel

# Crear el histograma y las curvas de kernel
fig, ax = plt.subplots(figsize=(12, 10))
for i in range(len(fun_rural_ld.columns)):
    sns.histplot(data=fun_rural_ld, x=fun_rural_ld.columns[i], color=kernel_color[i], multiple='dodge',
                 bins=int((max(fun_rural_ld.iloc[:, i]) - min(fun_rural_ld.iloc[:, i])) / bin_width), alpha=alpha, kde=False,
                 label=fun_rural_ld.columns[i], linewidth=2, ax=ax, stat='density')
    kde = sns.kdeplot(data=fun_rural_ld, x=fun_rural_ld.columns[i], bw_method='silverman', color=kernel_color[i], multiple='stack',
                      linestyle=kernel_style[i], linewidth=kernel_width[i], fill=True, alpha=0.5, label=fun_rural_ld.columns[i],
                      ax=ax)

# Configurar las leyendas y etiquetas del gráfico
plt.legend(title="Estaciones del SVCA", loc="upper right", labels=fun_rural_ld.columns, fontsize=12, frameon=True,
           facecolor='white', framealpha=0.8, markerscale=1)
plt.xlabel('LAeq, Day (dBA)', fontsize=17)
plt.ylabel('Frecuencia (Normalizada)', fontsize=17)
plt.title('HISTOGRAMA DE NIVELES DIURNOS DE RUIDO AERONÁUTICO ÁREA RURAL - FUNZA', fontsize=19, fontweight='bold')
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=10)

# Configurar el eje y secundario
ax2 = ax.twinx()
# ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
ax2.set_ylabel('Densidad de probabilidad de Kernel', fontsize=17)
ax2.grid(False)

# Configurar tamaño de ejes
min_value = fun_rural_ld.min()
max_value = fun_rural_ld.max()
min_value_fun_rural_ld = min_value.min()
max_value_fun_rural_ld = max_value.max()
plt.xlim(min_value_fun_rural_ld - 2, max_value_fun_rural_ld + 2)

# Mostrar el gráfico
# sns.despine()
plt.tight_layout()
plt.savefig('FUN_RURAL_LD.png')
plt.show()

"""FUNZA DIURNO AREA URBANA"""

fun_urb_ld=pd.concat([F24, F29],axis=1)
# Configuración de estilo
sns.set_style("whitegrid")
sns.set_palette("colorblind")
# Borrar datos NaN
fun_urb_ld = fun_urb_ld.dropna()
# Configurar los parámetros del histograma
bin_width = 1  # Ancho de los bins
alpha = 0.8  # Opacidad del histograma
density = True  # Normalizar el histograma

# Configurar los parámetros de las curvas de kernel
kernel_color = ['#FF5733', '#8E44AD', '#2980B9', '#F1C40F', '#D35400']  # Colores de las curvas de kernel
kernel_style = ['--', '-.', ':', '-.', ':']  # Estilos de las curvas de kernel
kernel_width = [2, 2, 2, 2, 2]  # Anchos de las curvas de kernel

# Crear el histograma y las curvas de kernel
fig, ax = plt.subplots(figsize=(12, 10))
for i in range(len(fun_urb_ld.columns)):
    sns.histplot(data=fun_urb_ld, x=fun_urb_ld.columns[i], color=kernel_color[i], multiple='dodge',
                 bins=int((max(fun_urb_ld.iloc[:, i]) - min(fun_urb_ld.iloc[:, i])) / bin_width), alpha=alpha, kde=False,
                 label=fun_urb_ld.columns[i], linewidth=2, ax=ax, stat='density')
    kde = sns.kdeplot(data=fun_urb_ld, x=fun_urb_ld.columns[i], bw_method='silverman', color=kernel_color[i], multiple='stack',
                      linestyle=kernel_style[i], linewidth=kernel_width[i], fill=True, alpha=0.5, label=fun_rural_ld.columns[i],
                      ax=ax)

# Configurar las leyendas y etiquetas del gráfico
plt.legend(title="Estaciones del SVCA", loc="upper right", labels=fun_urb_ld.columns, fontsize=12, frameon=True,
           facecolor='white', framealpha=0.8, markerscale=1)
plt.xlabel('LAeq, Day (dBA)', fontsize=17)
plt.ylabel('Frecuencia (Normalizada)', fontsize=17)
plt.title('HISTOGRAMA DE NIVELES DIURNOS DE RUIDO AERONÁUTICO ÁREA URBANA - FUNZA', fontsize=19, fontweight='bold')
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=10)

# Configurar el eje y secundario
ax2 = ax.twinx()
# ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
ax2.set_ylabel('Densidad de probabilidad de Kernel', fontsize=17)
ax2.grid(False)

# Configurar tamaño de ejes
min_value = fun_urb_ld.min()
max_value = fun_urb_ld.max()
min_value_fun_urb_ld = min_value.min()
max_value_fun_urb_ld = max_value.max()
plt.xlim(min_value_fun_urb_ld - 2, max_value_fun_urb_ld + 2)

# Mostrar el gráfico
# sns.despine()
plt.tight_layout()
plt.savefig('FUN_URB_LD.png')
plt.show()


"""SKBO DIURNO AREA INTERNA"""

skbo_ld=pd.concat([F13, F15, F19, F20],axis=1)
# Configuración de estilo
sns.set_style("whitegrid")
sns.set_palette("colorblind")
# Borrar datos NaN
skbo_ld = skbo_ld.dropna()
# Configurar los parámetros del histograma
bin_width = 1  # Ancho de los bins
alpha = 0.8  # Opacidad del histograma
density = True  # Normalizar el histograma

# Configurar los parámetros de las curvas de kernel
kernel_color = ['#FF5733', '#8E44AD', '#2980B9', '#F1C40F', '#D35400']  # Colores de las curvas de kernel
kernel_style = ['--', '-.', ':', '-.', ':']  # Estilos de las curvas de kernel
kernel_width = [2, 2, 2, 2, 2]  # Anchos de las curvas de kernel

# Crear el histograma y las curvas de kernel
fig, ax = plt.subplots(figsize=(12, 10))
for i in range(len(skbo_ld.columns)):
    sns.histplot(data=skbo_ld, x=skbo_ld.columns[i], color=kernel_color[i], multiple='dodge',
                 bins=int((max(skbo_ld.iloc[:, i]) - min(skbo_ld.iloc[:, i])) / bin_width), alpha=alpha, kde=False,
                 label=skbo_ld.columns[i], linewidth=2, ax=ax, stat='density')
    kde = sns.kdeplot(data=skbo_ld, x=skbo_ld.columns[i], bw_method='silverman', color=kernel_color[i], multiple='stack',
                      linestyle=kernel_style[i], linewidth=kernel_width[i], fill=True, alpha=0.5, label=skbo_ld.columns[i],
                      ax=ax)

# Configurar las leyendas y etiquetas del gráfico
plt.legend(title="Estaciones del SVCA", loc="upper right", labels=skbo_ld.columns, fontsize=12, frameon=True,
           facecolor='white', framealpha=0.8, markerscale=1)
plt.xlabel('LAeq, Day (dBA)', fontsize=17)
plt.ylabel('Frecuencia (Normalizada)', fontsize=17)
plt.title('HISTOGRAMA DE NIVELES DIURNOS DE RUIDO AERONÁUTICO - SKBO', fontsize=19, fontweight='bold')
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=10)

# Configurar el eje y secundario
ax2 = ax.twinx()
# ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
ax2.set_ylabel('Densidad de probabilidad de Kernel', fontsize=17)
ax2.grid(False)

# Configurar tamaño de ejes
min_value = skbo_ld.min()
max_value = skbo_ld.max()
min_value_skbo_ld = min_value.min()
max_value_skbo_ld = max_value.max()
plt.xlim(min_value_skbo_ld - 2, max_value_skbo_ld + 2)

# Mostrar el gráfico
# sns.despine()
plt.tight_layout()
plt.savefig('SKBO_LD.png')
plt.show()

"""SUBA"""

suba_ld=pd.concat([F34],axis=1)
# Configuración de estilo
sns.set_style("whitegrid")
sns.set_palette("colorblind")
# Borrar datos NaN
suba_ld = suba_ld.dropna()
# Configurar los parámetros del histograma
bin_width = 1  # Ancho de los bins
alpha = 0.8  # Opacidad del histograma
density = True  # Normalizar el histograma

# Configurar los parámetros de las curvas de kernel
kernel_color = ['#FF5733', '#8E44AD', '#2980B9', '#F1C40F', '#D35400']  # Colores de las curvas de kernel
kernel_style = ['--', '-.', ':', '-.', ':']  # Estilos de las curvas de kernel
kernel_width = [2, 2, 2, 2, 2]  # Anchos de las curvas de kernel

# Crear el histograma y las curvas de kernel
fig, ax = plt.subplots(figsize=(12, 10))
for i in range(len(suba_ld.columns)):
    sns.histplot(data=suba_ld, x=suba_ld.columns[i], color=kernel_color[i], multiple='dodge',
                 bins=int((max(suba_ld.iloc[:, i]) - min(suba_ld.iloc[:, i])) / bin_width), alpha=alpha, kde=False,
                 label=suba_ld.columns[i], linewidth=2, ax=ax, stat='density')
    kde = sns.kdeplot(data=suba_ld, x=suba_ld.columns[i], bw_method='silverman', color=kernel_color[i], multiple='stack',
                      linestyle=kernel_style[i], linewidth=kernel_width[i], fill=True, alpha=0.5, label=suba_ld.columns[i],
                      ax=ax)

# Configurar las leyendas y etiquetas del gráfico
plt.legend(title="Estaciones del SVCA", loc="upper right", labels=suba_ld.columns, fontsize=12, frameon=True,
           facecolor='white', framealpha=0.8, markerscale=1)
plt.xlabel('LAeq, Day (dBA)', fontsize=17)
plt.ylabel('Frecuencia (Normalizada)', fontsize=17)
plt.title('HISTOGRAMA DE NIVELES DIURNOS DE RUIDO AERONÁUTICO - SUBA', fontsize=19, fontweight='bold')
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=10)

# Configurar el eje y secundario
ax2 = ax.twinx()
# ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
ax2.set_ylabel('Densidad de probabilidad de Kernel', fontsize=17)
ax2.grid(False)

# Configurar tamaño de ejes
min_value = suba_ld.min()
max_value = suba_ld.max()
min_value_suba_ld = min_value.min()
max_value_suba_ld = max_value.max()
plt.xlim(min_value_suba_ld - 2, max_value_suba_ld + 2)

# Mostrar el gráfico
# sns.despine()
plt.tight_layout()
plt.savefig('SUBA_LD.png')
plt.show()

"""DIAGRAMA DE VIOLINES - LEQ - ENGATIVA"""

# sns.kdeplot(data=df_ln, x="LEVEL_LD", y="LEVEL_LD", hue="NMT", fill=True,) Grafica kernel en dos ejes (Prueba)
eng_leq_events=pd.concat([f1_leq_event,f2_leq_event,f10_leq_event,f18_leq_event,f33_leq_event],axis=1)
f, ax = plt.subplots()
sns.set_theme(style="whitegrid")
sns.violinplot(data=eng_leq_events, bw=0.5, inner="box")
# sns.despine(offset=10, trim=True) #Esta linea elimina la cuadricula de la grafica
ax.set_xlabel('Estaciones Localidad de Engativá',fontsize=17)
ax.set_ylabel('LEQ (dBA)',fontsize=17)
ax.set_title('NIVELES DE RUIDO EQUIVALENTES POR EVENTOS - DIAGRAMA DE VIOLINES - JORNADA DIURNA - ENGATIVÁ',fontsize=20,
             fontweight='bold')
plt.savefig('ENG_LEQ_LD.png')
plt.show()

"""DIAGRAMA DE VIOLINES - LEQ - FONTIBON"""
# sns.kdeplot(data=df_ln, x="LEVEL_LD", y="LEVEL_LD", hue="NMT", fill=True,) Grafica kernel en dos ejes (Prueba)
fon_leq_events=pd.concat([f3_leq_event,f7_leq_event,f21_leq_event,f23_leq_event,f25_leq_event,f27_leq_event,
                          f30_leq_event,f32_leq_event],axis=1)
f, ax = plt.subplots()
sns.set_theme(style="whitegrid")
sns.violinplot(data=fon_leq_events, bw=0.5, inner="box")
# sns.despine(offset=10, trim=True) #Esta linea elimina la cuadricula de la grafica
ax.set_xlabel('Estaciones Localidad de Fontibón',fontsize=17)
ax.set_ylabel('LEQ (dBA)',fontsize=17)
ax.set_title('NIVELES DE RUIDO EQUIVALENTES POR EVENTOS - DIAGRAMA DE VIOLINES - JORNADA DIURNA - FONTIBÓN',fontsize=20, fontweight='bold')
plt.savefig('FON_LEQ_LD.png')
plt.show()

"""DIAGRAMA DE VIOLINES - LEQ - FUNZA"""
# sns.kdeplot(data=df_ln, x="LEVEL_LD", y="LEVEL_LD", hue="NMT", fill=True,) Grafica kernel en dos ejes (Prueba)
fun_leq_events=pd.concat([f4_leq_event,f5_leq_event,f11_leq_event,f17_leq_event,f24_leq_event,f29_leq_event],axis=1)
f, ax = plt.subplots()
sns.set_theme(style="whitegrid")
sns.violinplot(data=fun_leq_events, bw=0.5, inner="box")
# sns.despine(offset=10, trim=True) #Esta linea elimina la cuadricula de la grafica
ax.set_xlabel('Estaciones Municipio de Funza',fontsize=17)
ax.set_ylabel('LEQ (dBA)',fontsize=17)
ax.set_title('NIVELES DE RUIDO EQUIVALENTES POR EVENTOS - DIAGRAMA DE VIOLINES - JORNADA DIURNA - FUNZA',fontsize=20,
             fontweight='bold')
plt.savefig('FUN_LEQ_LD.png')
plt.show()

"""DIAGRAMA DE VIOLINES - LEQ - SKBO"""
# sns.kdeplot(data=df_ln, x="LEVEL_LD", y="LEVEL_LD", hue="NMT", fill=True,) Grafica kernel en dos ejes (Prueba)
skbo_leq_events=pd.concat([f13_leq_event,f15_leq_event,f19_leq_event,f20_leq_event],axis=1)
f, ax = plt.subplots()
sns.set_theme(style="whitegrid")
sns.violinplot(data=skbo_leq_events, bw=0.5, inner="box")
# sns.despine(offset=10, trim=True) #Esta linea elimina la cuadricula de la grafica
ax.set_xlabel('Estaciones Aeropuerto Internacional El Dorado',fontsize=17)
ax.set_ylabel('LEQ (dBA)',fontsize=17)
ax.set_title('NIVELES DE RUIDO EQUIVALENTES POR EVENTOS - DIAGRAMA DE VIOLINES - JORNADA DIURNA - SKBO',fontsize=20,
             fontweight='bold')
plt.savefig('SKBO_LEQ_LD.png')
plt.show()

"""DIAGRAMA DE VIOLINES - LEQ - SUBA"""
# sns.kdeplot(data=df_ln, x="LEVEL_LD", y="LEVEL_LD", hue="NMT", fill=True,) Grafica kernel en dos ejes (Prueba)
suba_leq_events=pd.concat([f34_leq_event],axis=1)
f, ax = plt.subplots()
sns.set_theme(style="whitegrid")
sns.violinplot(data=suba_leq_events, bw=0.5, inner="box")
# sns.despine(offset=10, trim=True) #Esta linea elimina la cuadricula de la grafica
ax.set_xlabel('Estacion Localidad de Suba',fontsize=17)
ax.set_ylabel('LEQ (dBA)',fontsize=17)
ax.set_title('NIVELES DE RUIDO EQUIVALENTES POR EVENTOS - DIAGRAMA DE VIOLINES - JORNADA DIURNA - SUBA',fontsize=20,
             fontweight='bold')
plt.savefig('SUBA_LEQ_LD.png')
plt.show()

"""CREACIÓN DE .XLSX"""

r_aero=pd.ExcelWriter('Aeronautico_LD_Dic.xlsx')
df1=pd.DataFrame(df)

df.to_excel(r_aero,'Events')
df['FIS Aircraft type'].isna().sum()#Cuenta la cantidad de datos Nan en una columna esecifica
df_final.to_excel(r_aero,'LD_Hora')
diurno.to_excel(r_aero,'LD_Dia')
Count_Events.to_excel(r_aero,'Events_NMT')
freq_table.to_excel(r_aero, 'Freq_Table')
niv_air_type_perday.to_excel(r_aero, 'Nivel_type_day')
niv_air_type_permonth.to_excel(r_aero, 'Nivel_type_month')
niv_air_perday.to_excel(r_aero, 'Nivel_type_day_f')
niv_air_permonth.to_excel(r_aero, 'Nivel_type_month_f')
comp_linea_base.to_excel(r_aero,'Comp_LB')
comp_res0627.to_excel(r_aero,'Comp_Res0627')
top_niv_air_perday_max.to_excel(r_aero,'Lmax_airtype_day')
list_aircraft.to_excel(r_aero,'List_Aircraft')

r_aero.save()
r_aero.close()






"""Segundo tipo de grafico con libreria sns

iris = datasets.load_iris()

sns.kdeplot(df_ln.loc[(df_ln['NMT']=='F001'),
            'LEVEL_LD'], color='r', shade=True, Label='F001')

sns.kdeplot(df_ln.loc[(df_ln['NMT']=='F020'), 
            'LEVEL_LD'], color='b', shade=True, azLabel='F020')

plt.xlabel('LAeq, D (dBA)')
plt.ylabel('Frecuencia (Cantidad de días)')"""
