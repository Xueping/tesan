# Brett Kromkamp (brett@perfectlearn.com)
# You Programming (http://www.youprogramming.com)
# May 03, 2014

from src.utils.tree.node import Node

(_ROOT, _DEPTH, _BREADTH) = range(3)


class Tree:

    def __init__(self):
        self.__nodes = {}

    @property
    def nodes(self):
        return self.__nodes

    def add_node(self, identifier, parent=None):
        node = Node(identifier)
        self[identifier] = node

        if parent is not None:
            self[parent].add_child(identifier)

        return node

    def display(self, identifier, depth=_ROOT):
        children = self[identifier].children
        if depth == _ROOT:
            print("{0}".format(identifier))
        else:
            print("\t"*depth, "{0}".format(identifier))

        depth += 1
        for child in children:
            self.display(child, depth)  # recursive call

    def generate_json(self,json_dict, identifier, n_samples):
        children = self[identifier].children
        kids = []
        for child in children:
            kid = dict()
            kid["name"] = child
            if int(child) >= n_samples:
                kid["children"] = self.generate_json(kid, child, n_samples)  # recursive call
            else:
                kid["size"] = 1
            kids.append(kid)
        json_dict["children"] = kids
        return kids

    def generate_children(self,json_dict, identifier, n_samples):
        children = self[identifier].children
        kids = []
        for child in children:
            kid = dict()
            if int(child) >= n_samples:
                self.generate_children(kid, child, n_samples)  # recursive call
            else:
                kid["name"] = child
                kid["size"] = 1
            kids.append(kid)
        json_dict["children"] = kids
        return kids

    def traverse(self, identifier, mode=_DEPTH):
        # Python generator. Loosly based on an algorithm from
        # 'Essential LISP' by John R. Anderson, Albert T. Corbett,
        # and Brian J. Reiser, page 239-241
        yield identifier
        queue = self[identifier].children
        while queue:
            yield queue[0]
            expansion = self[queue[0]].children
            if mode == _DEPTH:
                queue = expansion + queue[1:]  # depth-first
            elif mode == _BREADTH:
                queue = queue[1:] + expansion  # width-first

    def __getitem__(self, key):
        return self.__nodes[key]

    def __setitem__(self, key, item):
        self.__nodes[key] = item