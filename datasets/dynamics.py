import numpy as np
# import networkx as nx

node_mapping = lambda G: {node: idx for idx, node in enumerate(G.nodes)}

def Model_Biochemical(xx, t, G, F = 1., B = 1., R = 1.):
    """
    m_0 = "F-B*xx[i]"
    m_1 = "xx[i]"
    m_2 = "R*xx[j]"
    """
    dxdt = []
    node_to_index = node_mapping(G)

    for node in G.nodes():
        i = node_to_index[node]
        m_0 = F - B * xx[i]
        m_1 = - R * xx[i]
        m_2 = sum([xx[node_to_index[neighbor]] for neighbor in G.neighbors(node)])
        dxdt.append(m_0 + m_1*m_2 )
        
    return np.array(dxdt)


def Model_Epidemics(xx, t, G, B = 1., R = 1.):
    
    dxdt = []
    node_to_index = node_mapping(G)
    
    for node in G.nodes():
        i = node_to_index[node]
        m_0 = -B * xx[i]
        m_1 = R * (1-xx[i])
        m_2 = sum([xx[node_to_index[neighbor]] for neighbor in G.neighbors(node)])
        dxdt.append(m_0 + m_1*m_2 )
        
    return np.array(dxdt)


def Model_Neuronal(xx, t, G, B = 1., C = 1., R = 1.):
    tan_xx = np.tanh(xx)
    dxdt = []
    node_to_index = node_mapping(G)
        
    for node in G.nodes():
        i = node_to_index[node]
        m_0 = -B * xx[i] + C * tan_xx[i]
        m_1 = R
        m_2 = sum([tan_xx[node_to_index[neighbor]] for neighbor in G.neighbors(node)])

        dxdt.append(m_0 + m_1*m_2 )
    return np.array(dxdt)


def Model_Kuramoto(xx, t, G, w_i, R=1.):
    dxdt = []
    node_to_index = node_mapping(G)
    
    for node in G.nodes():
        i = node_to_index[node]
        degree_i = len(list(G.neighbors(node)))
        interaction_sum = sum(
            np.sin(xx[node_to_index[neighbor]] - xx[i]) for neighbor in G.neighbors(node)
        )
        
        dxdt.append(w_i[i] + (R / degree_i) * interaction_sum)
        
    return np.array(dxdt)  

    