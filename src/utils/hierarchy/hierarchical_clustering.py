from scipy.cluster.hierarchy import linkage
import numpy as np
from heapq import heappush, heappushpop

def hc_get_descendent(node, children, n_leaves):
    ind = [node]
    # sub_children = []
    descendent = []

    if node < n_leaves:
        return ind

    n_indices = 1
    while n_indices:
        i = ind.pop()
        if i < n_leaves:
            descendent.append(i)
            n_indices -= 1
            # sub_children.append('null')
        else:
            ind.extend(children[i - n_leaves])
            # sub_children.append(children[i - n_leaves])
            n_indices += 1
    return descendent

def hc_cut(n_clusters, children, n_leaves):

    if n_clusters > n_leaves:
        raise ValueError('Cannot extract more clusters than samples: '
                         '%s clusters where given for a tree with %s leaves.'
                         % (n_clusters, n_leaves))

    nodes = [-(max(children[-1]) + 1)]

    nodes_pos = []
    nodes_leaves = []

    for i in range(n_clusters - 1):
        # As we have a heap, nodes[0] is the smallest element
        these_children = children[-nodes[0] - n_leaves]
        # Insert the 2 children and remove the largest node
        heappush(nodes, -these_children[0])
        heappushpop(nodes, -these_children[1])
    label = np.zeros(n_leaves, dtype=np.intp)
    for i, node in enumerate(nodes):
        nodes_pos.append(-node)
        # print(hc_get_descendent(-node, children, n_leaves))
        sub_leaves = hc_get_descendent(-node, children, n_leaves)
        nodes_leaves.append(sub_leaves)
        label[sub_leaves] = i
    return label, nodes_pos,nodes_leaves

def get_leaves(leaves,labels):
    children_list = []
    for leaf in leaves:
        leaf_dict = dict()
        leaf_dict["name"] = str(leaf)
        leaf_dict["level"] = 'Leaf'
        leaf_dict["size"] = 1
        leaf_dict['label'] = int(labels[leaf])
        children_list.append(leaf_dict)
    return children_list

def second_last_level(root_nodes, nodes_leaves, labels,level):
    children_list = []
    for i, node in enumerate(root_nodes):
        node_dict = dict()
        node_dict["name"] = 'Group_'+str(i+1)
        node_dict["level"] = 'Level_'+str(level)
        node_dict['children'] = get_leaves(nodes_leaves[i],labels)
        children_list.append(node_dict)
    return children_list

def middle_level(max_level, n_clusters,data,level=0):
    children_list = []
    n_leaves = data.shape[0]
    print(n_leaves)
    if n_leaves > 1:
        Z = linkage(data, 'ward')
        children = Z[:, [0, 1]].astype(np.int32)
        labels, root_nodes, nodes_leaves = hc_cut(n_clusters, children, n_leaves)
        if level > max_level - 1:
            return second_last_level(root_nodes, nodes_leaves, labels, level)
        level += 1
        for i, node in enumerate(root_nodes):
            # current root
            node_dict = dict()
            if level == 1:
                node_dict["name"] = 'Cluster_' + str(i + 1)+'_Level_' + str(level)
            else:
                node_dict["name"] = 'Level_' + str(level)
            node_dict["level"] = 'Level_' + str(level)
            # if len(nodes_children[i]) < 2:
            current_data = data[nodes_leaves[i]]
            if current_data.shape[0] < n_clusters:
                node_dict['children'] = get_leaves(nodes_leaves[i], labels)
            else:
                node_dict['children'] = middle_level(max_level,n_clusters,current_data,level)
            children_list.append(node_dict)
    else:
        node_dict = dict()
        node_dict["name"] = 'leaf'
        node_dict["level"] = 'leaf'
        node_dict['size'] = 1
        children_list.append(node_dict)

    return children_list