# coding=utf-8
"""
.. moduleauthor:: Shuhua Gao

This module :mod:`visualization` provides utility functions to visualization the expression tree from a given
K-expression, a gene or a chromosome in GEP.
"""

from ..core.entity import KExpression, Chromosome, Gene, GeneNN
from ..core.symbol import Function, Terminal, NN_Function_With_Jumpers


def _graph_kexpression(expr, starting_index):
    """
    Create a graph for a K-expression *expr* with the node's number starting from *starting_index*.

    :param expr: k-expression
    :param starting_index: the first number of nodes in the expression tree
    :return: A node list, an edge list, and a dictionary of labels.
    """
    assert len(expr) > 0
    nodes = [starting_index + i for i in range(len(expr))]
    edges = []
    labels = {}
    for i, p in enumerate(expr):
        if isinstance(p, Function):
            labels[starting_index + i] = p.name
        elif isinstance(p, Terminal):
            labels[starting_index + i] = p.format()
        else:
            raise RuntimeError('Unrecognized symbol. Normally, a symbol in the K-expression is either a function '
                               'or a terminal')
    i = 0
    j = 0
    while i < len(expr):
        for _ in range(expr[i].arity):
            j += 1
            edges.append((i + starting_index, j + starting_index))
        i += 1
    return nodes, edges, labels

# ------------------------ Ryan Heminway addition (start) ----------------- #
def _graph_kexpression_nn(expr, starting_index, weights, depths):
    """
    Create a graph for KExpression describing a GeneNN *expr* with the node's number starting from 
    *starting_index*.

    :param expr: a KExpression describing a GeneNN
    :param starting_index: the first number of nodes in the expression tree
    :param weights: array corresponding to weights of edges in nn graph
    :param depths: a list matching size of individual, describing depth of element node at each index
    :return: A node list, an edge list, and a dictionary of labels.
    """
    assert len(expr) > 0
    nodes = [] # [starting_index + i for i in range(len(expr))]
    edges = []
    input_nodes = {}
    labels = {}
    for i, p in enumerate(expr):
        if isinstance(p, Function):
            nodes.append(starting_index + i)
            labels[starting_index + i] = [p.name, depths[i]]
        elif isinstance(p, Terminal):
            name = p.format()
            if name not in input_nodes:
                nodes.append(starting_index + i)
                input_nodes[name] = starting_index + i
                labels[starting_index + i] = [name, depths[i]]
        else:
            raise RuntimeError('Unrecognized symbol. Normally, a symbol in the K-expression is either a function '
                               'or a terminal')
    i = 0
    j = 0
    last_j = 0
    while i < len(expr):
        element = expr[i]
        this_element_index = starting_index + i
        
        # Draw jumper edges
        if isinstance(element, NN_Function_With_Jumpers):
            for (from_element_index, weight) in element.get_forward_jumper_tuples():
                from_element = expr[from_element_index]
                if isinstance(from_element, NN_Function_With_Jumpers) and (depths[from_element_index] > i):
                    print("Visualizing a jumper!")
                    edges.append((from_element_index + starting_index, this_element_index, weight, "fwd"))
                    
            for (from_element_index, weight) in element.get_recurrent_jumper_tuples():
                from_element = expr[from_element_index]
                if isinstance(from_element, NN_Function_With_Jumpers):
                    print("Visualizing a jumper!")
                    edges.append((from_element_index + starting_index, this_element_index, weight, "rec"))
        
        
        # (NOTE Ryan) Indexing here is tough... normally weights are distributed right-to-left but its opposite here
        for k in range(expr[i].arity):
            j += 1
            p = expr[j]
            # Make input_nodes unique
            if isinstance(p, Terminal):
                name = p.format()
                # i + starting_index, input_nodes[name]
                edges.append((input_nodes[name], this_element_index, weights[len(weights) - (len(expr) - 1) + last_j + (expr[i].arity - k - 1)], "normal"))
            else:
                edges.append((j + starting_index, this_element_index, weights[len(weights) - (len(expr) - 1) + last_j + (expr[i].arity - k - 1)], "normal"))
        i += 1
        last_j = j
    return nodes, edges, labels

    
def graph_nn(genome, label_renaming_map=None):
    """
    Construct the graph of a genome. It returns in order a node list, an edge list, and a dictionary of the per node
    labels. The node are represented by numbers, the edges are tuples connecting two nodes (number), and the labels are
    values of a dictionary for which keys are the node numbers.

    :param genome: :class:`~geppy.core.entity.KExpression`, :class:`~geppy.core.entity.Gene`, or
        :class:`~geppy.core.entity.Chromosome`, the genotype of an individual
    :param label_renaming_map: dict, which maps the old name of a primitive (or a linking function)
        to a new one for better visualization. The default label for each node is just the name of the primitive
        placed on this node. For example, you may provide ``renamed_labels={'and_': 'and'}``.
    :return: A node list, an edge list, and a dictionary of labels.

    You can visualize a genome and export the tree visualization to an image file directly using the
    :func:`export_expression_tree` function.
    """
    nodes = []
    edges = []
    labels = {}
    
    def graph_gene_nn(genome, index):
        weights = [genome.dw_rnc_array[x] for x in genome.dw] 
        nodes, edges, labels = _graph_kexpression_nn(genome.kexpression, index, weights, genome.__find_depths__())
        return nodes, edges, labels
    
    if isinstance(genome, GeneNN):
        nodes, edges, labels = graph_gene_nn(genome, 0)
    elif isinstance(genome, Chromosome):
        if len(genome) == 1:
            nodes, edges, labels = graph_gene_nn(genome[0], 0)
        else:   # multigenic chromosome, we need to concatenate multiple trees
            starting_index = 1
            sub_roots = []
            for gene in genome:
                expr = gene.kexpression
                sub_roots.append(starting_index)
                sub_nodes, sub_edges, sub_labels = graph_gene_nn(gene, starting_index)
                nodes.extend(sub_nodes)
                edges.extend(sub_edges)
                labels.update(sub_labels)
                starting_index += len(expr)
            # connect subtrees by inserting the linker node as 0
            nodes.append(0)
            for root in sub_roots:
                edges.append((root, 0, 1, "normal")) # (NOTE Ryan) Assuming linking functions use weights of 1
            labels[0] = [genome.linker.__name__, 0] # (TODO Ryan) is depth 0 right here?
    else:
        raise TypeError('Only an argument of type KExpression, Gene, and Chromosome is acceptable. The provided '
                        'genome type is {}.'.format(type(genome)))
    # rename_labels labels
    if label_renaming_map is not None:
        for k, v in labels.items():
            if v[0] in label_renaming_map:
                labels[k][0] = label_renaming_map[v[0]]
    return nodes, edges, labels

def export_expression_tree_nn(genome, label_renaming_map=None, file='tree.png'):
    """
    Construct the graph of a *genome* containing GeneNNs and then export it to a *file*.

    :param genome: :class:`~geppy.core.entity.GeneNN`, or 
        :class:`~geppy.core.entity.Chromosome`, the genotype of an individual
    :param label_renaming_map: dict, which maps the old name of a primitive (or a linking function)
        to a new one for better visualization. The default label for each node is just the name of the primitive
        placed on this node. For example, you may provide ``renamed_labels={'and_': 'and'}``.
    :param file: str, the file path to draw the expression tree, which may be a relative or absolute one.
        If no extension is included in *file*, then the default extension 'png' is used.

    .. note::
        This function currently depends on the :mod:`graphviz` module to render the tree. Please first install the
        `graphviz <https://pypi.org/project/graphviz/>`_ module before using this function.
        Alternatively, you can always obtain the raw graph data with the :func:`graph` function, then postprocess the
        data and render them with other tools as you want.
    """
    import graphviz as gv
    import os.path

    _, edges, labels = graph_nn(genome, label_renaming_map)
    file_name, ext = os.path.splitext(file)
    ext = ext.lstrip('.')
    # Make 2 graphs, one with labels
    g = gv.Digraph(format=ext, engine='dot', graph_attr={'splines': 'ortho'})
    g2 = gv.Digraph(format=ext, engine='dot', graph_attr={'splines': 'ortho'})
    sorted_nodes = sorted(labels.items(), key=lambda x: x[1][1])
    
    for name, label in sorted_nodes:
        g.node(str(name), str(label[0]))  # add node
        g2.node(str(name), str(label[0]))
        
    # (TODO) This code is messy... was prototyping
    # Structure nodes by depth
    # Find the minimum and maximum depths
    min_depth = 0
    max_depth = max(label[1] for name, label in sorted_nodes)
    
    for name, label in sorted_nodes:
        if label[1] == -1:
            label[1] = max_depth + 1
    max_depth = max_depth + 1
    
    # Structure the nodes into rows by depth using rank
    for depth in range(min_depth, max_depth + 1):
        with g.subgraph() as s:
            s.attr(rank='same')
            for name, label in sorted_nodes:
                if label[1] == depth:
                    s.node(str(name), str(label[0]))
    
    for name1, name2, label, edge_type in edges:
        if edge_type is "fwd":
            color = "green"
        elif edge_type is "rec":
            color = "red"
        else:
            color = "black"
        g.edge(str(name1), str(name2), xlabel="", dir="Forward", color=color)  # add edge
        g2.edge(str(name1), str(name2), xlabel=str(label), dir="Forward", color=color)  # add edge
    g.render(file_name)    
    file_name_labels = file_name + "_labeled"
    g2.render(file_name_labels)
# ------------------------ Ryan Heminway addition (end) ----------------- #


def graph(genome, label_renaming_map=None):
    """
    Construct the graph of a genome. It returns in order a node list, an edge list, and a dictionary of the per node
    labels. The node are represented by numbers, the edges are tuples connecting two nodes (number), and the labels are
    values of a dictionary for which keys are the node numbers.

    :param genome: :class:`~geppy.core.entity.KExpression`, :class:`~geppy.core.entity.Gene`, or
        :class:`~geppy.core.entity.Chromosome`, the genotype of an individual
    :param label_renaming_map: dict, which maps the old name of a primitive (or a linking function)
        to a new one for better visualization. The default label for each node is just the name of the primitive
        placed on this node. For example, you may provide ``renamed_labels={'and_': 'and'}``.
    :return: A node list, an edge list, and a dictionary of labels.

    You can visualize a genome and export the tree visualization to an image file directly using the
    :func:`export_expression_tree` function.
    """
    nodes = []
    edges = []
    labels = {}
    if isinstance(genome, KExpression):
        nodes, edges, labels = _graph_kexpression(genome, 0)
    elif isinstance(genome, Gene):
        nodes, edges, labels = _graph_kexpression(genome.kexpression, 0)
    elif isinstance(genome, Chromosome):
        if len(genome) == 1:
            nodes, edges, labels = _graph_kexpression(genome[0].kexpression, 0)
        else:   # multigenic chromosome, we need to concatenate multiple trees
            starting_index = 1
            sub_roots = []
            for gene in genome:
                expr = gene.kexpression
                sub_roots.append(starting_index)
                sub_nodes, sub_edges, sub_labels = _graph_kexpression(
                    expr, starting_index)
                nodes.extend(sub_nodes)
                edges.extend(sub_edges)
                labels.update(sub_labels)
                starting_index += len(expr)
            # connect subtrees by inserting the linker node as 0
            nodes.append(0)
            for root in sub_roots:
                edges.append((0, root))
            labels[0] = genome.linker.__name__
    else:
        raise TypeError('Only an argument of type KExpression, Gene, and Chromosome is acceptable. The provided '
                        'genome type is {}.'.format(type(genome)))
    # rename_labels labels
    if label_renaming_map is not None:
        for k, v in labels.items():
            if v in label_renaming_map:
                labels[k] = label_renaming_map[v]
    return nodes, edges, labels

def export_expression_tree(genome, label_renaming_map=None, file='tree.png'):
    """
    Construct the graph of a *genome* and then export it to a *file*.

    :param genome: :class:`~geppy.core.entity.KExpression`, :class:`~geppy.core.entity.Gene`, or
        :class:`~geppy.core.entity.Chromosome`, the genotype of an individual
    :param label_renaming_map: dict, which maps the old name of a primitive (or a linking function)
        to a new one for better visualization. The default label for each node is just the name of the primitive
        placed on this node. For example, you may provide ``renamed_labels={'and_': 'and'}``.
    :param file: str, the file path to draw the expression tree, which may be a relative or absolute one.
        If no extension is included in *file*, then the default extension 'png' is used.

    .. note::
        This function currently depends on the :mod:`graphviz` module to render the tree. Please first install the
        `graphviz <https://pypi.org/project/graphviz/>`_ module before using this function.
        Alternatively, you can always obtain the raw graph data with the :func:`graph` function, then postprocess the
        data and render them with other tools as you want.
    """
    import graphviz as gv
    import os.path

    _, edges, labels = graph(genome, label_renaming_map)
    file_name, ext = os.path.splitext(file)
    ext = ext.lstrip('.')
    g = gv.Graph(format=ext)
    for name, label in labels.items():
        g.node(str(name), str(label))  # add node
    for name1, name2, label in edges:
        g.edge(str(name1), str(name2), label=str(label))  # add edge
    g.render(file_name)
    


__all__ = ['graph', 'export_expression_tree', 'export_expression_tree_nn']
