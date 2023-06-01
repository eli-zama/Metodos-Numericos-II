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




st.title(':purple[Método de Quasi-Newton(BFGS)]')

st.subheader(':purple[Descripción del método]')
st.write(''':purple[Este método hace uso tanto del gradiente como de una aproximación a la inversa de la matriz Hessiana 
        de la función, esto para hacer una aproximación al cálculo de la segunda derivada. Por ser un método de aproximación
         a la segunda derivada se dice que es un método Quasi-Newtoniano. \nEl problema con el método es el costo computacional
         para funciones de muchas variables, ya que se almacena una matriz cuadrada de datos tan grande como el cuadrado de la
         cantidad de variables. \nEs adecuado para funciones no lineales de varias variables y la búsqueda del óptimo sin 
         restricciones. Si bien el método hace que la convergencia al óptimo sea rápida por ser de aproximación a la 
         segunda derivada, el costo de procesamiento es bastante alto.]''')

st.subheader(':purple[Ejemplo]')

st.subheader('Método')
xxs = st.text_input('Ingrese la función $f(x)$: ',value='(x - 2)**2 + (y - 3.95)**2')



fx = sy.parse_expr(xxs)
intstrr = ''


st.latex('f'+str(tuple(fx.free_symbols))+' = '+sy.latex(fx))
if len(fx.free_symbols)<= 2:
    if len(fx.free_symbols) == 1:
        func = sy.lambdify(list(fx.free_symbols),fx)
        plo = gro.Figure()
        plo.add_trace(gro.Scatter(x=np.linspace(-10,10,1000),y=func(np.linspace(-10,10,1000))))
        st.plotly_chart(plo)
        p =sy.plot(fx,show=False)
        pl = get_sympy_subplots(p)

        st.pyplot(pl)

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



initaprx = st.text_input('Ingrese una aproximacion inicial $x_0$: ',value='[0,1.5]')

intaprox = []
intstr = ''




for i in initaprx:

    if i != '{' and i != '}' and i != '[' and i != ']' and i != '(' and i != ')' and i != ' ':
        intstr = intstr + i

try:
    st.write('La aproximacion inicial es: ')
    intaprox = list(map(int, intstr.split(',')))
    st.latex(sy.latex(sy.Matrix(list(intaprox))))
except:
    st.error('Error al introducir la aproximación inicial', icon="🚨")

err = st.text_input('Ingrese el error de tolerancia: ',value='0.00001')
try:
    st.write('El error de tolerancia es:', float(err))
except:
    st.error('Error al introducir el error de tolerancia', icon="🚨")


maxiter = st.slider('Maximo de Iteraciones',10,1000,10)




#COLOCA TU METODO AQUI y PASA LA  FUNCION ALOJADA EN fx