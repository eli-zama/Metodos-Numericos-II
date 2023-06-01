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


def discrete_minimun_quads_aprox(xs,y,functs,symbs):

    m = []


    for i in range(0,len(xs)):
        aux = []

        for j in range(0,len(functs)):

            aux.append(functs[j])
        m.append(aux)


    #pprint(Matrix(m))

    mev = []
    for i in range(0,len(m)):
        aux = []

        for j in range(0,len(m[0])):
            if len(m[i][j].free_symbols) > 0:
                aux.append(m[i][j].subs(symbs,xs[i]))
            else:
                aux.append(m[i][j])
        mev.append(aux)

    #pprint(Matrix(mev))

    mevT = sy.Matrix(mev).transpose()
    #pprint(mevT)

    a = mevT*sy.Matrix(mev)

    #pprint(a)

    b = mevT*sy.Matrix(y)

    #pprint(b)

    ainv = a.inv()

    xsol = ainv*b

    #pprint(xsol)


    expr = xsol[0]+xsol[1]*symbs


    p = sy.plot(expr,show=False)
    p2 = get_sympy_subplots(p)

    p2.plot(xs,y,"o")
    #p2.show()
    return sy.expand(expr),p2


st.title(':purple[Aproximación discreta de minimos cuadrados]')

st.subheader(':purple[Descripción del método]')
st.write(''':purple[El método de los mínimos cuadrados se utiliza para calcular la recta de regresión lineal que minimiza 
        los residuos, esto es, las diferencias entre los valores reales y los estimados por la recta. Se revisa su 
        fundamento y la forma de calcular los coeficientes de regresión con este método. \nEl modelo de regresión lineal 
        posibilita, una vez establecida una función lineal, efectuar predicciones sobre el valor de una variable $Y$ 
        sabiendo los valores de un conjunto de variables $X1, X2,… Xn$. A la variable $Y$ la llamamos dependiente, 
        aunque también se la conoce como variable objetivo, endógena, criterio o explicada. Por su parte, las variables 
        $X$ son las variables independientes, conocidas también como predictoras, explicativas, exógenas o regresoras.]''')
st.subheader(':purple[Ejemplo]')



xxs = st.text_input('Ingrese los valores de $x_n$: ',value='{-1,2,5,8}')

xsstr = ''


for i in xxs:

    if i != '{' and i != '}' and i != '[' and i != ']' and i != '(' and i != ')' and i != ' ':
        xsstr = xsstr + i

fxxs = st.text_input('Ingrese los valores de $f(x_n)$: ',value='{3,1,8,5}')

x = list(map(float,xsstr.split(',')))
intstrr = ''




for t in fxxs:

    if t != '{' and t != '}' and t != '[' and t != ']' and t != '(' and t != ')' and t != ' ':
        intstrr = intstrr + t

fx = list(map(float,intstrr.split(',')))



funx = st.text_input('Ingrese las funciones $f_k(x)$ a considerar:',value='{1,x**2}')
funcstr = ''

for t in funx:

    if t != '{' and t != '}' and t != '[' and t != ']' and t != '(' and t != ')' and t != ' ':
        funcstr = funcstr +t

#st.write(funcstr)
funcs = []
for i in funcstr.split(','):
    funcs.append(sy.parse_expr(i,transformations='all'))

sym = list(funcs[0].free_symbols)

l = 0
while l < len(funcs):
    if len(funcs[l].free_symbols) != 0:
        sym = list(funcs[l].free_symbols)
        break
    l += 1

#st.write(str(sym))
method = discrete_minimun_quads_aprox(x,fx,funcs,sym[0])

st.write('La combinacion lineal que mejor se ajusta a los datos es:')
st.latex('f(x)='+sy.latex(method[0]))


func = sy.lambdify(sym[0],method[0])

plo = gro.Figure()
plo.add_trace(gro.Scatter(x=x,y=fx,name='Datos'))
plo.add_trace(gro.Scatter(x=np.linspace(min(x)-10,max(x)+10,1000),y=func(np.linspace(min(x)-10,max(x)+10,1000)),name='Aproximación'))

st.plotly_chart(plo)