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




st.title(':purple[Integración Numérica)]')

st.subheader(':purple[Descripción del método]')
st.write(''':purple[Dada una función f definida sobre un intervalo [a,b], estamos interesados en calcular $J(f) = \int_{a}^{b} f(x) dx$
        suponiendo que esta integral tenga sentido para la función f. La cuadratura o integración numérica 
        consiste en obtener fórmulas aproximadas para calcular la integral $J(f)$ de $f$. \n
        Estos métodos son de gran utilidad cuando la integral no se puede calcular por métodos analíticos, su cálculo 
        resulta muy costoso y estamos interesados en una solución con precisión finita dada o bien sólo disponemos de 
        una tabla de valores de la función (es decir, no conocemos la forma analítica de f).]''')

st.subheader(':purple[Ejemplo]')
st.subheader('Método')
xxs = st.text_input('Ingrese la función $f(x)$: ',value='exp(-2x)')



fx = sy.parse_expr(xxs,transformations='all')
intstrr = ''


fxxs = st.text_input('Ingrese el intervalo $[a,b]$: ',value='[1,3]')


for t in fxxs:

    if t != '{' and t != '}' and t != '[' and t != ']' and t != '(' and t != ')' and t != ' ':
        intstrr = intstrr + t

interval = list(map(float,intstrr.split(',')))

symbs = list(fx.free_symbols)

dxx = ''
integ = sy.integrate(fx, symbs)
for i in symbs:
    dxx = dxx +'d'+ str(i)

st.latex(r'\int '+sy.latex(fx)+dxx+' = '+sy.latex(integ)+' + C')

st.latex(r'\int_{'+str(interval[0])+'}^{'+str(interval[1])+'}'+sy.latex(fx)+dxx)

if len(fx.free_symbols)<= 2:
    if len(fx.free_symbols) == 1:
        func = sy.lambdify(list(fx.free_symbols),fx)
        plo = gro.Figure()
        plo.add_trace(gro.Scatter(x=np.linspace(interval[0],interval[1],1000),y=func(np.linspace(interval[0],interval[1],1000)),fill='tozeroy'))
        plo.update_layout(title='Integral de f'+str(tuple(fx.free_symbols))+' en el intervalo '+str(interval))
        st.plotly_chart(plo)


    if  len(fx.free_symbols) == 2:
        func = sy.lambdify(list(fx.free_symbols),fx)
        plo = gro.Figure()
        ran = np.linspace(-10,10,100)
        su = [[func(ran[xs],ran[ys]) for xs in range (0,len(ran)) ] for ys in range(0,len(ran))]
        plo.add_trace(gro.Surface(z=su))
        st.plotly_chart(plo)
        p =plot3d(fx,show=False)
        pl = get_sympy_subplots(p)

        st.pyplot(pl)


#COLOCA TU METODO AQUI y PASA LA  FUNCION ALOJADA EN fx
