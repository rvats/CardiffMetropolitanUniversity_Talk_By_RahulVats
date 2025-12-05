from pylab import plt
import numpy as np
import networkx as nx

def partition_to_draw(partition_obj):
    if type(partition_obj == 'dict'):
        partition = np.array([partition_obj[i] for i in range(len(partition_obj))])
    
    count = 0.
    list_nodes = []
    color      = []
    for com in set(partition_obj):
        count = count + 1.

        this_com_nodes = []
        for i in range(len(partition_obj)):      
            if partition_obj[i] == com:
                this_com_nodes.append(i)

        list_nodes.extend(this_com_nodes)

        color.extend([count] * len(this_com_nodes))

    color = np.array(color)
    
    return list_nodes, color
    


def draw_partitioned_graph(G, partition_obj, layout=None, labels=None,layout_type='spring', 
               node_size=70, node_alpha=0.7, cmap=plt.get_cmap('jet'),
               node_text_size=12,
               edge_color='blue', edge_alpha=0.5, edge_tickness=1,
               edge_text_pos=0.3,
               text_font='sans-serif'):

    # if a premade layout haven't been passed, create a new one
    if not layout:
        if graph_type == 'spring':
            layout=nx.spring_layout(G)
        elif graph_type == 'spectral':
            layout=nx.spectral_layout(G)
        elif graph_type == 'random':
            layout=nx.random_layout(G)
        else:
            layout=nx.shell_layout(G)

    # prepare the partition list noeds and colors

    list_nodes, node_color = partition_to_draw(partition_obj)
      
    # draw graph
    nx.draw_networkx_nodes(G,layout,list_nodes,node_size=node_size, 
                           alpha=node_alpha, node_color=node_color, cmap = cmap)
    nx.draw_networkx_edges(G,layout,width=edge_tickness,
                           alpha=edge_alpha,edge_color=edge_color)
    #nx.draw_networkx_labels(G, layout,font_size=node_text_size,
    #                        font_family=text_font)

    if labels is None:
        labels = range(len(G))

    edge_labels = dict(zip(G, labels))
    #nx.draw_networkx_edge_labels(G, layout, edge_labels=edge_labels, 
    #                            label_pos=edge_text_pos)

    # show graph

    plt.axis('off')
    plt.xlim(0,1)
    plt.ylim(0,1)


