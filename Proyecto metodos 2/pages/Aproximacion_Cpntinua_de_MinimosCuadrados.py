import streamlit as st
import pandas as pd
import numpy as np
import sympy as sy
from matplotlib import pyplot as plt
from sympy.plotting.plot import MatplotlibBackend, Plot
from sympy.plotting import plot3d,plot3d_parametric_line
import plotly as ply
import plotly.express as ex
import plotly.graph_objects as gro
from plotly.subplots import make_subplots

def get_sympy_subplots(plot:Plot):
   
    backend = MatplotlibBackend(plot)

    backend.process_series()
    backend.fig.tight_layout()
    return backend.plt



def continium_minimun_quads_aprox(fx,interv,symb,ang):
    """
    Given a function, an interval, a symbol and a ang, it returns the polynomial that best approximates the function in
    the given interval
    :param fx: The function to be approximated
    :param interval: the interval in which the function is defined
    :param symb: the symbol that will be used to represent the variable in the function
    :param ang: The ang of the polynomial
    :return: The function that is the best aproximation of the given function in the given interv.
    """

    m = []


    for i in range(0,ang+1):
        aux = []
        for j in range(0,ang+1):
            aux.append(sy.integrate((symb**i)*(symb**j),(symb,interv[0],interv[1])))
        m.append(aux)

    #pprint(Matrix(m))


    b = []

    for i in range(0,ang+1):
        b.append(sy.integrate((symb**i)*fx,(symb,interv[0],interv[1])))

    #pprint(Matrix(b))

    sol = sy.Matrix(m).inv() * sy.Matrix(b)

    expr = 0

    for i in range(0,ang+1):
        expr = expr + (sol[i]*symb**i)

    #pprint(expr)


    p = sy.plot(fx,(symb,interv[0],interv[1]),show=False)
    p.append(sy.plot(expr,(symb,interv[0],interv[1]),show=False)[0])

    #p.show()


    return sy.expand(expr),get_sympy_subplots(p)

st.title(':purple[Aproximación continua de minimos cuadrados]')

st.subheader(':purple[Descripción del método]')
st.write(''':purple[Mínimos cuadrados es una técnica de análisis numérico enmarcada dentro de la optimización matemática, 
            en la que, dados un conjunto de pares ordenados y una familia de funciones, se intenta encontrar la función 
            continua, dentro de dicha familia, que mejor se aproxime a los datos, de acuerdo con el criterio de mínimo 
            error cuadrático. \nEn su forma más simple, intenta minimizar la suma de cuadrados de las diferencias en 
            las ordenadas (también llamadas residuos) entre los puntos generados por la función elegida y los 
            correspondientes valores en los datos. Específicamente, se llama mínimos cuadrados promedio (LMS) cuando el 
            número de datos medidos es 1 y se usa el método de descenso por gradiente para minimizar el residuo cuadrado]''')
st.subheader(':purple[Ejemplo]')


st.subheader('Método')
xxs = st.text_input('Ingrese la función $f(x)$: ',value='cos(pi*x)')



fx = sy.parse_expr(xxs,transformations='all')
intstrr = ''


fxxs = st.text_input('Ingrese el intervalo $[a,b]$: ',value='[-1,1]')


for t in fxxs:

    if t != '{' and t != '}' and t != '[' and t != ']' and t != '(' and t != ')' and t != ' ':
        intstrr = intstrr + t

interv = list(map(float,intstrr.split(',')))



ang = st.slider('Grado del polinomio de aproximación: ',1,10,value=2)

method = continium_minimun_quads_aprox(fx,interv,list(fx.free_symbols)[0],int(ang))

st.write('El polinomio esta dado por:')
st.latex('P_{'+str(ang)+'}(x)='+sy.latex(method[0]))




plo = gro.Figure()
func = sy.lambdify(list(fx.free_symbols)[0],fx)
aproxfunc = sy.lambdify(list(fx.free_symbols)[0],method[0])
plo.add_trace(gro.Scatter(x = np.linspace(interv[0],interv[1],1000),y=func(np.linspace(interv[0],interv[1],1000)),name='Función', marker_color='rgba(152, 0, 0, .8)'))
plo.add_trace(gro.Scatter(x=np.linspace(interv[0],interv[1],1000),y=aproxfunc(np.linspace(interv[0],interv[1],1000)),name='Aproximación',fill='tonexty'))
st.plotly_chart(plo)


st.subheader('Evaluador de a Aproximación: ')
evalu = st.number_input('Ingrese el punto a evaluar: ',value=0.5)

evv = method[0].subs({list(fx.free_symbols)[0]: evalu}).evalf()


st.latex('f('+str(sy.latex(evalu))+r') \approx '+sy.latex(evv))