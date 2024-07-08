# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 00:16:39 2024

@author: kingy
"""
#proyecto 1
# crear graficos rapidos  por mes 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo Excel
df = pd.read_excel('C:/Users/kingy/OneDrive/Escritorio/Recuperacion.xlsx')  # Reemplaza 'archivo.xlsx' con el nombre de tu archivo Excel

# Guardar el DataFrame en formato CSV
#df.to_csv('C:/Users/kingy/OneDrive/Escritorio/archivo.csv', index=False)  # Guarda el archivo CSV como 'archivo.csv'


datos= pd.read_csv('C:/Users/kingy/OneDrive/Escritorio/Recuperacion1.csv', skiprows=2, encoding='latin1', delimiter=',') # separacion por ;


df=pd.DataFrame(datos)



#quitar la primera columna 

df = df.iloc[:, 1:]   

df.info()

# Eliminar las filas que contienen NaN en la columna 'columna'
#df.dropna(subset=['columna'], inplace=True)
df.dropna(subset=['Rec Limpieza Cu'], inplace=True)

print(df.columns)


filtro = (df['Rec Limpieza Cu'] <= 100) 

#resultado = df[filtro]
resultado = df[filtro]

#hasta ahora tenemos una columna sin nan y con valores menores a 100%


# Utilizando el método drop() para eliminar la columna
#resultado.drop('mes_año', axis=1, inplace=True)



# Supongamos que 'fecha' es una columna de tipo datetime
resultado['Hora termino'] = pd.to_datetime(resultado['Hora termino'])

# Extraer el año y el mes y combinarlos en una nueva columna
resultado['mes_y_año'] = pd.to_datetime(resultado['Hora termino'].dt.year * 10000 + resultado['Hora termino'].dt.month * 100 + 1, format='%Y%m%d')

# 'format='%Y%m%d'' es opcional, pero ayuda a definir el formato de la fecha resultante.

# Supongamos que tienes un DataFrame 'df' con la columna 'fecha' que contiene fechas en formato datetime

# Crear una nueva columna con el formato "mes-año"
#resultado['mes_año1'] = resultado['mes_y_año'].dt.strftime('%B-%y') #mes en ingles no sirve



# Visualizar el DataFrame con la nueva columna
#print(df)


# Supongamos que tienes un DataFrame 'df' con la columna 'fecha' que contiene fechas en formato datetime

# Crear una nueva columna con el formato "mes-año" en español
meses = {
    1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril', 5: 'mayo', 6: 'junio', 
    7: 'julio', 8: 'agosto', 9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre'
}


resultado['mes_año'] = resultado['mes_y_año'].dt.month.map(meses) + '-' + resultado['mes_y_año'].dt.strftime('%y')


mapeo_texto = {
    'enero-23':'01.enero-23',
    'febrero-23':'02.febrero-23',
    'marzo-23': '03.marzo-23',
    'abril-23': '04.abril-23',
    'mayo-23': '05.mayo-23',
    'junio-23': '06.junio-23',
    'julio-23': '07.julio-23',
    'agosto-23': '08.agosto-23',
    'septiembre-23': '09.septiembre-23',
    'octubre-23': '10.octubre-23',
    'noviembre-23': '11.noviembre-23',
    'diciembre-23': '12.diciembre-23',
    'enero-24': '13.enero-24',
    # Agrega más mapeos según sea necesario
}

# Utiliza la función map() para aplicar el mapeo a la columna 'texto' y crear una nueva columna 'valor_recodificado'
resultado['valor_recodificado'] = resultado['mes_año'].map(mapeo_texto)





"""

# Crear una figura y ejes
fig, ax = plt.subplots()
# Graficar un boxplot de la variable por mes
resultado.boxplot(column='Rec Limpieza Cu', by='valor_recodificado', ax=ax)

# Configuración adicional de la gráfica
ax.set_xlabel('Mes')
ax.set_ylabel('Variable a graficar')
ax.set_title('Boxplot de la variable por mes')

# Rotar etiquetas del eje x verticalmente
plt.xticks(rotation=90)  # Ajusta el ángulo de rotación según sea necesario
# Mostrar la gráfica
plt.show()


"""








#Crear el boxplot (se ejecuta el bloque completo) lo muestra sin outliners, mediana con solo 
# 1 decimal, hay que mejorar el formato, tiene el eje x girado, 

# Configurar el color de fondo de la figura
plt.rcParams['figure.facecolor'] = '#dce1e3'

#bp = resultado.boxplot(column='Rec Limpieza Cu', by='valor_recodificado',  showfliers=False)
bp = resultado.boxplot(column='Rec Limpieza Cu', by='valor_recodificado',  showfliers=False,  patch_artist=True, boxprops=dict(facecolor='#0276a8'), color='black')


# Calcular la mediana para cada grupo
medians = resultado.groupby('valor_recodificado')['Rec Limpieza Cu'].median()

# Agregar la mediana al gráfico
#for i, median in enumerate(medians):
#    bp.text(i + 1, median, f'{median:.1f}', horizontalalignment='center', verticalalignment='bottom', color='black')
    

for i, mediana in enumerate(medians, start=1):
    plt.plot(i, mediana, marker='o', markersize=5, color='black')
    bp.text(i, mediana, f'{mediana:.1f}', horizontalalignment='center', verticalalignment='bottom', color='black')


# Calcular la media para cada grupo
#means = resultado.groupby('valor_recodificado')['Rec Limpieza Cu'].median()

# Crear un DataFrame para la línea media
#dfm_mean = pd.DataFrame({'variable': means.index, 'value': means.values})

# Plot a line plot with markers for the means
#sns.lineplot(data=dfm_mean, x='variable', y='value', marker='o', legend=False)


# Conectar las medianas con una línea media


X = list(range(1, len(medians) + 1))
Y = medians.values
plt.plot(X, Y, marker='o', markersize=5, color='black')


# Eliminar el segundo título generado automáticamente
plt.suptitle('')
# Establecer el límite máximo del eje y en 100
#plt.ylim(70, 100)

# Mostrar el boxplot con la mediana
plt.title('Recuperacion de limpieza de Cu')
plt.xlabel('Mes')
plt.ylabel('Recuperacion de limpieza (%)')
# Rotar etiquetas del eje x verticalmente
plt.xticks(rotation=90)  # Ajusta el ángulo de rotación según sea necesario
plt.show()

######
######
######
#chequear  este estilo
######
######
######

import seaborn as sns

# Establecer el estilo de Seaborn
#sns.set_style("darkgrid")

# Lista completa de estilos
sns.axes_style()

parametros = {"axes.edgecolor": "red", "grid.linestyle": "dashed", "grid.color": "black"}
sns.set_style("darkgrid", rc = parametros)




plt.figure(figsize=(6,6))
sns.boxplot(x=resultado['Turno'],y=resultado['Rec. Cu L1 Quimico'], color='skyblue', palette=None)
plt.axvline(x=1.5, color='red', linestyle='--') #linea vertical

# Calcular la posición para colocar el comentario
x_pos = 0.5
y_pos = df['Rec. Cu L1 Quimico'].max() * 0.9
# Agregar un comentario
plt.text(x_pos, y_pos, 'Comentario', fontsize=12, ha='center', va='bottom', color='red')
plt.title('X')
plt.show()


#darkgrid: Fondo oscuro con líneas de cuadrícula blancas.
#whitegrid: Fondo claro con líneas de cuadrícula blancas.
#dark: Fondo oscuro sin líneas de cuadrícula.
#white: Fondo claro sin líneas de cuadrícula.
#ticks: Fondo claro con marcas de eje en forma de ticks.

import pandas as pd
import matplotlib.pyplot as plt

# Crear un DataFrame de ejemplo
data = {
    'Rec Limpieza Cu': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'valor_recodificado': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
}
resultado = pd.DataFrame(data)

# Extraer los datos para cada grupo 'valor_recodificado'
grupo1 = resultado[resultado['valor_recodificado'] == 1]['Rec Limpieza Cu']
grupo2 = resultado[resultado['valor_recodificado'] == 2]['Rec Limpieza Cu']

# Configurar el gráfico de boxplot con color azul en la caja
plt.boxplot([grupo1, grupo2], showmeans=True, meanline=True, patch_artist=True, boxprops=dict(facecolor='blue'))

# Configuración adicional del gráfico
plt.title('Boxplot con Mediana (sin Outliers)')
plt.xlabel('Grupo')
plt.ylabel('Rec Limpieza Cu')
plt.xticks([1, 2], ['Grupo 1', 'Grupo 2'])

# Mostrar el boxplot
plt.show()










