import numpy as np
from scipy.integrate import solve_ivp

# Mapping nodes to indices for array use
node_mapping = lambda G: {node: idx for idx, node in enumerate(G.nodes)}

dynamics_name = ['Biochemical', 'Epidemics', 'Neuronal', 'Kuramoto', 'Population']


def Model_Biochemical(t, xx, G, F=1., B=1., R=1.):
    dxdt = np.zeros_like(xx)
    node_to_index = node_mapping(G)
    for node in G.nodes():
        i = node_to_index[node]
        m_0 = F - B * xx[i]
        m_1 = - R * xx[i]
        m_2 = sum([xx[node_to_index[neighbor]] for neighbor in G.neighbors(node)])
        dxdt[i] = m_0 + m_1 * m_2
    return dxdt


def Model_Epidemics(t, xx, G, B=1., R=1.):
    dxdt = np.zeros_like(xx)
    node_to_index = node_mapping(G)
    for node in G.nodes():
        i = node_to_index[node]
        m_0 = -B * xx[i]
        m_1 = R * (1 - xx[i])
        m_2 = sum([xx[node_to_index[neighbor]] for neighbor in G.neighbors(node)])
        dxdt[i] = m_0 + m_1 * m_2
    return dxdt


def Model_Neuronal(t, xx, G, B=1., C=1., R=1.):
    tan_xx = np.tanh(xx)
    dxdt = np.zeros_like(xx)
    node_to_index = node_mapping(G)
    for node in G.nodes():
        i = node_to_index[node]
        m_0 = -B * xx[i] + C * tan_xx[i]
        m_1 = R
        m_2 = sum([tan_xx[node_to_index[neighbor]] for neighbor in G.neighbors(node)])
        dxdt[i] = m_0 + m_1 * m_2
    return dxdt


def Model_Kuramoto(t, xx, G, w=0., R=1.):
    dxdt = np.zeros_like(xx)
    node_to_index = node_mapping(G)
    for node in G.nodes():
        i = node_to_index[node]
        interaction_sum = sum(
            [np.sin(xx[node_to_index[neighbor]] - xx[i]) for neighbor in G.neighbors(node)]
        )
        dxdt[i] = w + R * interaction_sum
    return dxdt


def Model_Population(t, xx, G, B=1., R=1., b=2, a=2):
    xx_powa = np.power(xx, a)
    xx_powb = np.power(xx, b)
    dxdt = np.zeros_like(xx)
    node_to_index = node_mapping(G)
    for node in G.nodes():
        i = node_to_index[node]
        m_0 = -B * xx_powb[i]
        m_1 = R
        m_2 = sum([xx_powa[node_to_index[neighbor]] for neighbor in G.neighbors(node)])
        dxdt[i] = m_0 + m_1 * m_2
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
    elif dynamics == 'Population':
        model = lambda t, xx: Model_Population(t, xx, G, **kwargs)
    else:
        raise Exception('Not supported dynamics!')

    t_eval = np.linspace(time_span[0], time_span[1], t_eval_steps)
    sol = solve_ivp(model, time_span, initial_state, t_eval=t_eval, method='RK45')

    return sol.y.T, sol.t  # shape: (t_steps, n_nodes), (t_steps,)
    