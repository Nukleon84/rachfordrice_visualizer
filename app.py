from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

EPS_T = 1e-15
MAX_ITR = 50
delta=1e-4

def rachford_rice_solver(
    Nc: int, zi: np.ndarray, Ki: np.ndarray, V0:np.float64, check_bounds:bool
) -> Tuple[int, np.ndarray, np.ndarray, float, float]:
    """
    This function solves for the root of the Rachford-Rice equation. The solution
    is defined by the contestent and this is the only part of the code that the
    contestent should change.
    The input is the number of components (Nc), the total composition (zi), and the K-values (KI).
    The output is the number of iterations used (N), the vapor molar composition (yi), the liquid
    molar composition (xi), the vapor molar fraction (V), and the liquid molar fraction (L).
    """
    K_max = np.max(Ki)
    K_min = np.min(Ki)
    V_min = 1 / (1 - K_max)
    V_max = 1 / (1 - K_min)

    def rr(V: float) -> float:
        return np.dot(zi, (Ki - 1) / (1 + V * (Ki - 1)))

    def drr(V: float) -> float:
        return -np.dot(zi, ((Ki - 1) / (1 + V * (Ki - 1))) ** 2)
      
    V=V0
    h = np.inf

    Niter = 0
    Vh=[V0]
    hh=[rr(V0)]

    while abs(h) > EPS_T:
        Niter += 1

        
        h = rr(V)
        dh = drr(V)
      

        if h > 0:
            V_min = V
        else:
            V_max = V

        V = V - h / dh

        if check_bounds:
            if V < V_min or V > V_max:
                V = (V_max + V_min) / 2

        Vh.append(V)
        hh.append(0)
        Vh.append(V)
        hh.append(rr(V))

        if Niter > MAX_ITR:
            break

    L = 1 - V
    xi = zi / (1 + V * (Ki - 1))
    yi = Ki * xi

    return Niter, yi, xi, V, L, Vh, hh



def plot_rr(Nc: int, zi: np.ndarray, Ki: np.ndarray,V0:np.float64, show_newton,show_ln_bounds, check_bounds):
        
    def rr(V: float) -> float:
        return np.dot(zi, (Ki - 1) / (1 + V * (Ki - 1)))

    K_max = np.max(Ki)
    K_min = np.min(Ki)
    V_min = 1 / (1 - K_max)
    V_max = 1 / (1 - K_min)

    #V= np.linspace(V_min+EPS_T,V_max-EPS_T, 1000)
    V= np.linspace(-10, V_min-delta, 500)
    y=list(map(rr,V))
        
    fig=go.Figure()
    fig.add_scatter(    
            x=V,
            y=y,
            name="Rachford-Rice (left asymptote)",
            line=dict(color="blue")
    )
    V= np.linspace(V_min+delta, V_max-delta, 500)
    y=list(map(rr,V))
      
    
    fig.add_scatter(    
            x=V,
            y=y,
            name="Rachford-Rice ",
            line=dict(color="blue")
    )
    V= np.linspace(V_max+delta, 2*V_max, 500)
    y=list(map(rr,V))
        
    
    fig.add_scatter(    
            x=V,
            y=y,
            name="Rachford-Rice (right asymptote)",
            line=dict(color="blue")
    )

    (niter, yi, xi, V,L, iterV, iterH)=rachford_rice_solver(Nc, zi, Ki, V0,check_bounds)
    
    col1, col2,col3 = st.columns(3)
    col1.metric(f"Iterations",niter)
    col2.metric(f"Vapor Fraction",V)
    col3.metric(f"Function Value",rr(V))
    
    if(show_newton):
        fig.add_scatter(    
                x=iterV,
                y=iterH,
                name="Newton-Path",
                line=dict(color="red")
                
        )
   

    #fig.add_vline(x=V, line_color="magenta",annotation_text="V_solution")
   
    fig.add_vline(x=V_min,line_dash="dash", line_color="green",annotation_text="V_min")
    fig.add_vline(x=V_max,line_dash="dash", line_color="green",annotation_text="V_max")
    if(show_ln_bounds):
        ln_min= (Ki[0]*z[0]-1)/(Ki[0]-1)
        ln_max= (1-z[-1])/(1-Ki[-1])
        fig.add_vline(x=ln_min,line_dash="dash", line_color="orange",annotation_text="LN_min")
        fig.add_vline(x=ln_max,line_dash="dash", line_color="orange",annotation_text="LN_max")

    fig.update_layout(        
        width=600,
        height=600,
        title="Rachford-Rice",
        xaxis_title="Vapor Fraction",
        yaxis_title="Residual Value of the Rachford-Rice equation",
        legend_title="Legend",
        font=dict(            
            size=12,            
        )
    )
    fig.update_xaxes(showgrid=True, zeroline=True, zerolinecolor="#444", zerolinewidth=2,showline=True, range=(V_min-1, V_max+1),titlefont = dict(size=20))
    fig.update_yaxes(showgrid=True, zeroline=True, zerolinecolor="#444", zerolinewidth=2,showline=True, range=(-2, 2),titlefont = dict(size=20))
 


    return fig

st.set_page_config(layout="wide")
st.title("Rachford-Rice Interactive Analyzer")


with st.sidebar:
    st.header("Problem Statement")

    NC= st.slider("Number of component",2,4,2)
    st.subheader("Compositions")
    n=[]
    z=[]
    K=[]
    for i in range(NC):
        ni=st.slider(f"Amount of Component {i+1} (mol)",delta,100.0, 10.0,1e-2)
        n.append(ni)

   
    st.subheader("Constant K-Values")
    for i in range(NC):
        if(i>0 and i != NC-1):
            Ki= st.slider(f"K-Value of Component {i+1}",delta, K[i-1]-1e-1, 1.4)
        elif (i == NC-1):
            Ki= st.slider(f"K-Value of Component {i+1}",delta, 1.0, 0.5)
        else:
            Ki= st.slider(f"K-Value of Component {i+1}",1.0, 50.0, 1.4)
        K.append(Ki)
    
    #z =np.array([zL, 1-zL])
    #Ki=np.array([KL, KH])
    #st.subheader("Intermediate Results")
    for i in range(NC):
        zi=n[i]/sum(n)
        z.append(zi)
        #st.write(f"Molar fraction of component {i+1}: {zi}")

    K_max = np.max(K)
    K_min = np.min(K)
    V_min = 1 / (1 - K_max)
    V_max = 1 / (1 - K_min)

    st.subheader("Options")
    show_newton=st.checkbox("Show Newton iterations", True)
    show_ln_bounds=st.checkbox("Show Leibovici-Neoschil bounds", False)
    check_bounds=st.checkbox("Do Bounds-checking", True)

    init=st.checkbox("User-defined Initial Value", True)
    if(init):
        V0=st.slider("V0",float(V_min),float(V_max), 0.5)
    else:
        V0=(V_min+V_max)/2
   
    
fig= plot_rr(int(NC), np.array(z), np.array(K), V0, show_newton,show_ln_bounds, check_bounds)
st.plotly_chart(fig, theme="streamlit", use_container_width=True)

