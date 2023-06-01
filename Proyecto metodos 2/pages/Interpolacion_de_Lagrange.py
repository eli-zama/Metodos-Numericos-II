import streamlit as st
import pandas as pd
import numpy as np
import sympy as sy
from matplotlib import pyplot as plt
from sympy.plotting.plot import MatplotlibBackend, Plot
from sympy.plotting import plot3d,plot3d_parametric_line
import plotly.express as ex
import plotly.graph_objects as gro
from plotly.subplots import make_subplots


def get_sympy_subplots(plot:Plot):
   
    backend = MatplotlibBackend(plot)

    backend.process_series()
    backend.fig.tight_layout()
    return backend.plt

def li(v, i):
 
    x = sy.symbols('x')

    s = 1
    st = ''
    for k in range(0,len(v)):
        if k != i:
            st = st + '((' + str(x) + '-'+ str(v[k])+')/('+str(v[i])+'-'+str(v[k])+'))'
            s = s*((x-v[k])/(v[i]-v[k]))

    return s

def Lagrange(v,fx):
 
    #print(v)
    #print(fx)
    lis = []
    for i in range(0,len(v)):
        lis.append(li(v,i))

    sums = 0

    for k in range(0,len(v)):
        sums = sums+(fx[k]*lis[k])

    #print(sums)

    sy.simplify(sums)

    sy.pprint(sums)

    p1 = sy.plot(sums,show=False)
    p2 = get_sympy_subplots(p1)
    p2.plot(v,fx,"o")
    #p2.show()
    return sy.expand(sums), p2,lis

st.title(':purple[Interpolación de Lagrange]')

st.subheader(':purple[Descripción del método]')
st.write(''':purple[Un polinomio de interpolación de Lagrange, p, se define en la forma:\n$p(x) = y_{0}\ell_{0}(x) + y_{1}\ell_{1}(x) + \cdots + y_{n}\ell_{n}(x) = \sum_{k=0}^{n} y_{k}\ell_{k}(x)$
            En donde $\ell_{0}, \ell_{1}, \dots, \ell_{n}$ son polinomios que dependen sólo de los nodos tabulados $x_{0},x_{1},\dots,x_{n}$, pero no de las ordenadas $y_{0},y_{1},\dots,y_{n}$. 
            \nLa fórmula general del polinomio $\ell_{i}$ es:\n
            $\ell_{i}(x) = \prod_{j=0, j \neq i}^{n} \frac{x-x_{j}}{x_{i}-x_{j}}$
            \nPara el conjunto de nodos $x_{0},x_{1},\dots,x_{n}$, estos polinomios son conocidos como funciones cardinales.
              Utilizando estos polinomios en la ecuación obtenemos la forma exacta del polinomio de interpolación de Lagrange.]''')
st.subheader(':purple[Ejemplo]')


st.subheader('Método')

filess = st.sidebar.file_uploader('Selecciona un archivo de prueba: ')
if filess != None:
    fi = pd.read_csv(filess)
    st.write('Los datos a interpolar son: ')
    st.write(fi)
    x = list(fi['x'])
    fx = list(fi['y'])
else:
    xxs = st.text_input('Ingrese los valores de $x_k$: ',value='{1,3,5,7}')

    xsstr = ''


    for i in xxs:

        if i != '{' and i != '}' and i != '[' and i != ']' and i != '(' and i != ')' and i != ' ':
            xsstr = xsstr + i

    fxxs = st.text_input('Ingrese los valores de $f(x_k)$: ',value='{-1,3,5,7}')

    x = list(map(float,xsstr.split(',')))
    intstrr = ''




    for t in fxxs:

        if t != '{' and t != '}' and t != '[' and t != ']' and t != '(' and t != ')' and t != ' ':
            intstrr = intstrr + t

    fx = list(map(float,intstrr.split(',')))


#st.write(x)
#st.write(fx)
#data = [x,fx]
#st.write(data)


method = Lagrange(x,fx)

st.write('_Los polinomios fundamentales de Lagrange estan dados por:_')
lli = r'''l_i(x) = \begin{cases}'''
for t in range(0,len(method[2])):
    lli = lli +'l_'+str(t)+r'='+sy.latex(sy.expand(method[2][t]))+r'\\'
lli = lli + r'\end{cases}'
st.latex(lli)
st.write('_El polinomio de Interpolacion está dado por:_')
st.latex(r'p_n(x) = \sum_{i=0}^{n} l_i(x)f(x_i)')
st.latex('p_n(x) =' + sy.latex(method[0]))

func = sy.lambdify(sy.symbols('x'),method[0])
funcdata = pd.DataFrame(dict(x=np.linspace(-10,10,1000),y=func(np.linspace(-10,10,1000))))

plo = gro.Figure()

plo.add_trace(gro.Scatter(x=np.linspace(-10,10,1000),y=func(np.linspace(-10,10,1000)),name='Interpolación'))
plo.add_trace(gro.Scatter(x=x,y=fx, marker_color='rgba(152, 0, 0, .8)',name='Datos'))
#plo.add_hline(y=0)
#plo.add_vline(x=0)
plo.update_layout(title='Grafica de la Interpolación')
st.plotly_chart(plo)