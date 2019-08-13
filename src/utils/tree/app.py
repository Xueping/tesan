import itertools
from src.utils.tree.tree import Tree

def build_tree(n_leaves, children):
    ii = itertools.count(n_leaves)
    non_leaf = {str(next(ii)): [str(x[0]), str(x[1])] for x in children}

    key_list = list(non_leaf.keys())
    key_list.reverse()
    root = key_list.pop(0)

    (_ROOT, _DEPTH, _BREADTH) = range(3)

    tree = Tree()
    tree.add_node(root)  # root node
    tree.add_node(non_leaf[root][0], root)
    tree.add_node(non_leaf[root][1], root)
    for key in key_list:
        child = non_leaf[key]
        tree.add_node(child[0], key)
        tree.add_node(child[1], key)

    return tree

# data = pd.read_csv('dataset/features_rep.csv')
# n_clusters = 4
# clustering = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
# clustering.fit(data)
# labels = clustering.labels_
# children = clustering.children_
#
# Z = linkage(data, 'ward')
# children = Z[:,[0,1]].astype(np.int32)
# labels, root_nodes, nodes_children, nodes_leaves = hc_cut(n_clusters, children, n_leaves)
#
# print(labels)
# print(root_nodes)
# print(nodes_children)
# print(nodes_leaves)
# tree = build_tree(n_leaves,children)
# tree.display('741')
# tree.generate_json(json_dict, display_root, n_leaves)
# tree.generate_children(json_dict, display_root, n_leaves)