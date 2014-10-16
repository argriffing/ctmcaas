"""
Data and tree for the MG94 gene conversion example.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx
from numpy.testing import assert_equal

__all__ = ['ad_hoc_fasta_reader', 'get_tree_info_with_outgroup']


def ad_hoc_fasta_reader(fin):
    name_seq_pairs = []
    while True:

        # read the name
        line = fin.readline().strip()
        if not line:
            return name_seq_pairs
        assert_equal(line[0], '>')
        name = line[1:].strip()

        # read the single line sequence
        line = fin.readline().strip()
        seq = line
        unrecognized = set(line) - set('ACGT')
        if unrecognized:
            raise Exception('unrecognized nucleotides: ' + str(unrecognized))

        name_seq_pairs.append((name, seq))


def get_tree_info_with_outgroup():
    T = nx.DiGraph()
    root = 'N0'
    T.add_edges_from([
            ('N0', 'kluyveri'),
            ('N0', 'N1'),
            ('N1', 'castellii'),
            ('N1', 'N2'),
            ('N2', 'bayanus'),
            ('N2', 'N3'),
            ('N3', 'kudriavzevii'),
            ('N3', 'N4'),
            ('N4', 'mikatae'),
            ('N4', 'N5'),
            ('N5', 'paradoxus'),
            ('N5', 'cerevisiae'),
            ])
    return T, root
