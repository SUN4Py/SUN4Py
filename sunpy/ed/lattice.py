#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Lattice:
    """
    Class to represent a geometrical lattice
    """
    
    def __init__(self):
        """
        Constructor of empty Lattice object
        """
        self._Ns = int(0)
        self._links = []
        self._nlinks = int(0)
        return
    
    def add_link(self, i, j):
        """
        Add an interaction link (i--j) to the lattice
        
        Remarks
        -------
        - The routine does not check if the link has already been added to the 
          list of links
        """
        minij = min(i, j)
        maxij = max(i, j)
        if (minij<maxij):
            self._links.append(np.array([minij, maxij], dtype=int))
            self._nlinks += 1
        self._Ns = max(self._Ns, maxij+1)
        return
    
    def plot(self):
        """
        Plot the graph of links
        """
        G = nx.Graph()
        G.add_edges_from(self._links)
        plt.figure(figsize=(8, 6))
        nx.draw(G, with_labels=True, node_color='lightgreen', 
                node_size=500, font_size=16, font_weight='bold')
        plt.title(f'Graph of lattice links (max distance: {self.max_distance()}; total distance= {np.sum(self.get_distances())})')
        plt.show()
        return
    
    def optimize(self):
        """
        Attempt to optimize the graph by renumbering nodes in order to minimize
        the distance bewteen nodes
        """
        
        G = nx.Graph()
        G.add_edges_from(self._links)
        '''
        #------------------------
        # METHOD 1
        #------------------------
        new_ordering = list(nx.cuthill_mckee_ordering(G))
        #------------------------
        '''
        
        #------------------------
        # METHOD 2
        #------------------------
        start_node = min(G.nodes(), key=lambda x: G.degree(x))
        new_ordering = list(nx.bfs_tree(G, start_node).nodes())
        #------------------------
        
        # map nodes
        old_to_new = {old_node: new_node for new_node, old_node in enumerate(new_ordering)}
        # Relabel the graph
        G_reordered = nx.relabel_nodes(G, old_to_new)
        # Get new links
        new_links = list(G_reordered.edges())        
        
        self._links = new_links
        return
    
    def max_distance(self):
        """
        Returns the maximal distance between two nodes of a link
        """
        max_dist = int(0)
        for link in self._links:
            max_dist = max(max_dist, abs(link[0]-link[1]))
        return max_dist
    
    def get_distances(self):
        """
        Returns all distances bewteen nodes of links
        """
        distances = np.zeros(self._nlinks, dtype=int)
        for i, link in enumerate(self._links):
            distances[i] = abs(link[0]-link[1])
        distances = np.sort(distances)
        return distances
    
    @property
    def Ns(self):
        return self._Ns
    
    @property
    def nlinks(self):
        return self._nlinks
    
    @property
    def links(self):
        return self._links
    

class chainLattice(Lattice):
    """
    Chain lattice
    """
    
    def __init__(self, Ns, isPBC):
        """
        Constructor of a chain lattice with Ns sites
        
        Parameters
        ----------
        Ns : int
            number of sites
        isPBC : bool
            True for periodic boundary conditions
            False for open boundary conditions
        """
        super().__init__()
        self._Ns = int(Ns)
        if isPBC==False:
            for i in range(0, self._Ns-1):
                self._links.append(np.array([i, i+1], dtype=int))
        else:            
            self._links.append(np.array([0, 1], dtype=int))
            for i in range(0, self._Ns-1):
                self._links.append(np.array([i, min(i+2, self._Ns-1)], dtype=int))
        self._nlinks = len(self._links)
        return
