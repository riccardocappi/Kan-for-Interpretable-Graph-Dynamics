import numpy as np
from scipy.integrate import solve_ivp
from torchdiffeq import odeint
import torch

import numpy as np
# import networkx as nx

node_mapping = lambda G: {node: idx for idx, node in enumerate(G.nodes)}

dynamics_name = ['Biochemical', 'Epidemics', 'Neuronal', 'Kuramoto']


def Model_Biochemical(t, xx, G, F = 1., B = 1., R = 1.):
    """
    m_0 = "F-B*xx[i]"
    m_1 = "xx[i]"
    m_2 = "R*xx[j]"
    """
    dxdt = torch.zeros_like(xx)
    node_to_index = node_mapping(G)

    for node in G.nodes():
        i = node_to_index[node]
        m_0 = F - B * xx[i]
        m_1 = - R * xx[i]
        m_2 = sum([xx[node_to_index[neighbor]] for neighbor in G.neighbors(node)])
        dxdt[i] = m_0 + m_1*m_2
        
        
    return dxdt


def Model_Epidemics(t, xx, G, B = 1., R = 1.):
    
    dxdt = torch.zeros_like(xx)
    node_to_index = node_mapping(G)
    
    for node in G.nodes():
        i = node_to_index[node]
        m_0 = -B * xx[i]
        m_1 = R * (1-xx[i])
        m_2 = sum([xx[node_to_index[neighbor]] for neighbor in G.neighbors(node)])
        dxdt[i] = m_0 + m_1*m_2
        
    return dxdt


def Model_Neuronal(t, xx, G, B = 1., C = 1., R = 1.):
    tan_xx = torch.tanh(xx)
    dxdt = torch.zeros_like(xx)
    node_to_index = node_mapping(G)
        
    for node in G.nodes():
        i = node_to_index[node]
        m_0 = -B * xx[i] + C * tan_xx[i]
        m_1 = R
        m_2 = sum([tan_xx[node_to_index[neighbor]] for neighbor in G.neighbors(node)])

        dxdt[i] = m_0 + m_1*m_2
    return dxdt


def Model_Kuramoto(t, xx, G, w=0., R=1.):
    dxdt = torch.zeros_like(xx)
    node_to_index = node_mapping(G)
    
    for node in G.nodes():
        i = node_to_index[node]
        degree_i = len(list(G.neighbors(node)))
        interaction_sum = sum(
            [torch.sin(xx[node_to_index[neighbor]] - xx[i]) for neighbor in G.neighbors(node)]
        )
        dxdt[i] = w + R * interaction_sum
        
    return dxdt  



def numerical_integration(G, dynamics, initial_state, time_span, t_eval_steps=100, **kwargs):
    assert dynamics in dynamics_name
    if dynamics == 'Biochemical':
        model = lambda t, xx: Model_Biochemical(t, xx, G, **kwargs)
    elif dynamics == 'Epidemics':
        model = lambda t, xx: Model_Epidemics(t, xx, G, **kwargs)
    elif dynamics == 'Neuronal':
        model = lambda t, xx: Model_Neuronal(t, xx, G, **kwargs)
    elif dynamics == 'Kuramoto':
        model = lambda t, xx: Model_Kuramoto(t, xx, G, **kwargs)
    else:
        raise Exception('Not supported dynamics!')

    t_eval = torch.linspace(time_span[0], time_span[1], t_eval_steps, device=initial_state.device)
    ys = odeint(model, initial_state, t_eval, method='dopri5')
    
    return ys, t_eval
    