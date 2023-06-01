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




st.title(':purple[Derivación Numérica]')

st.subheader(':purple[Descripción del método]')
st.write(''':purple[Esta es una técnica de análisis numérico para calcular una aproximación a la derivada de una función 
            en un punto utilizando los valores y propiedades de la misma. \nLa principal idea que subyace en las técnicas
         de derivación numérica está muy vinculada a la interpolación y se podría resumir en lo siguiente: 
         \nSi de una función $f(x)$ se conocen sus valores en un determinado soporte de puntos, puede "aproximarse” la 
         función $f(x)$ por otra función $p(x)$ que la interpole en dicho soporte y sustituir el valor de las derivadas de
         $f(x)$ en un punto $x*$ por el valor de las correspondientes derivadas de $p(x)$ en dicho punto $x*$.]''')

st.subheader(':purple[Ejemplo]')
st.subheader('Método')
xxs = st.text_input('Ingrese la función $f(x)$: ',value='(x - 5)**3')



fx = sy.parse_expr(xxs,transformations='all')
intstrr = ''

st.latex(r'\frac{\partial f}{\partial x}'+sy.latex(fx))


if len(fx.free_symbols)<= 2:
    if len(fx.free_symbols) == 1:
        func = sy.lambdify(list(fx.free_symbols),fx)
        plo = gro.Figure()
        plo.add_trace(gro.Scatter(x=np.linspace(-10,10,1000),y=func(np.linspace(-10,10,1000))))
        st.plotly_chart(plo)
        p =sy.plot(fx,show=False)
        pl = get_sympy_subplots(p)



    if  len(fx.free_symbols) == 2:
        func = sy.lambdify(list(fx.free_symbols),fx)
        plo = gro.Figure()
        ran = np.linspace(-10,10,100)
        su = [[func(ran[xs],ran[ys]) for xs in range (0,len(ran)) ] for ys in range(0,len(ran))]
        plo.add_trace(gro.Surface(z=su))
        st.plotly_chart(plo)
        p =plot3d(fx,show=False)
        pl = get_sympy_subplots(p)




#COLOCA TU METODO AQUI y PASA LA  FUNCION ALOJADA EN fx