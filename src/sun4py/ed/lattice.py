"""
SUN4Py A Python Library for solving SU(N) Heisenberg models
Copyright (C) 2026  Samuel Gozel, GNU GPLv3

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class Lattice:
    """
    Class to represent a lattice. 
    
    In SUNPy, a lattice does not carry a geometrical meaning. A lattice has 
    rather a Hamiltonian meaning, as it is defined as a collection of interaction 
    bonds, an interaction bond being an object describing an interaction
    term of the Hamiltonian between two different sites. A lattice assumes a 
    fixed, well-defined numbering of sites, as given by the definition of its 
    interaction bonds. It is left to the user to define a bijective map between 
    coordinates in real space on their geometrical lattice onto integers running
    from 0 to Ns-1 (included), Ns being the total number of sites in their/the
    lattice.
    """
    
    def __init__(self):
        """
        Constructor of empty Lattice object
        """
        self._Ns = int(0)
        self._bonds = []
        self._nbonds = int(0)
        self._unique_links = np.zeros(shape=(0, 2), dtype=int)
        self._indices = []
        self._bond_orders = []
        self._n_unique_links = int(0)
        return
    
    def add_bond(self, i, j, bond_type='HB', J=1.0):
        """
        Add an interaction bond (i--j) to the lattice
        
        Parameters
        ----------
        i : int
            first site involved in the bond
        j : int
            second site involved in the bond
        bond_type : str [optional][default: 'HB']
            type of interaction on the bond. Possible values are 'HB', 'HB2', ...
        J : float [optional][default: 1.0]
            coupling constant
        """
        newbond = Bond(i, j, bond_type, J)
        if not self.contains(newbond):
            self._bonds.append(newbond)
            self._nbonds += 1
            self._Ns = max(self._Ns, max(i, j)+1)
            
            newlink = newbond.bond
            ind = np.argwhere(np.sum(abs(self._unique_links - newlink), axis=1)==0).flatten()
            if len(ind)==0:
                self._unique_links = np.vstack((self._unique_links, newlink))
                self._indices.append(np.array([self._nbonds-1], dtype=int))
                self._bond_orders.append(np.array([newbond.bond_order], dtype=int))
                self._n_unique_links += 1
            else:
                ind = ind[0]
                self._indices[ind] = np.hstack((self._indices[ind], self._nbonds-1)).flatten()
                self._bond_orders[ind] = np.hstack((self._bond_orders[ind], newbond.bond_order)).flatten()
                sortinds = np.argsort(self._bond_orders[ind])
                self._indices[ind] = self._indices[ind][sortinds]
                self._bond_orders[ind] = self._bond_orders[ind][sortinds]
            
        else:
            sys.exit('ERROR : Lattice.add_bond : bond already exists.')
        return
    
    def contains(self, bond):
        """
        Check if the bond is already contained in the Lattice
        
        Returns
        -------
        isin : bool
            True if bond belongs to the Lattice, False otherwise
        """
        isin = False
        for bd in self._bonds:
            if ((bd.site1==bond.site1) & (bd.site2==bond.site2) & 
                (bd.bond_type==bond.bond_type)):
                isin = True
                break
        return isin
    
    def contains_link(self, link):
        """
        Check if the link has already been added on a previous bond
        """
        isin = False
        for lk in self._unique_links:
            if ((lk[0]==link[0]) & (lk[1]==link[1])):
                isin = True
                break
        return isin
    
    def set_coupling(self, i, j, bond_type, J):
        """
        Set the coupling of the bond between sites i and j to value J
        
        Parameters
        ----------
        i : int
            first site involved in the bond
        j : int
            second site involved in the bond
        bond_type : str
            type of interaction on the bond. Possible values are 'HB', 'HB2', ...
        J : float
            coupling constant
        """
        if bond_type=='HB':
            bond_type = 'HB1'
        found = False
        for bond in self._bonds:
            if ((bond.site1==min(i, j)) & (bond.site2==max(i, j)) & (bond.bond_type==bond_type)):
                bond.J = J
                found = True
                break
        assert(found)
        return
    
    def _get_links(self, bond_type):
        """
        
        """
        links = []
        for bond in self._bonds:
            if bond.bond_type==bond_type:
                newlink = [bond.site1, bond.site2]
                if not newlink in links:
                    links.append(newlink)
        return links
    
    def print(self):
        """
        Print each bond to screen
        """
        for bond in self._bonds:
            print(bond)
    
    def plot(self):
        """
        Plot the graph of links
        """
        all_bond_types = ['HB1', 'HB2', 'HB3', 'HB4']
        colors = ['k', 'r', 'b', 'm']
        
        G = nx.Graph()
        edge_list = [[] for _ in range(len(all_bond_types))]
        edge_colors = [[] for _ in range(len(all_bond_types))]
        
        bond_types = []
        bond_type_indices = []
        
        for i, bond_type in enumerate(all_bond_types):
            links = self._get_links(bond_type)
            if len(links):
                bond_types.append(bond_type)
                bond_type_indices.append(i)
                for link in links:
                    G.add_edge(link[0], link[1])
                    edge_list[i].append((link[0], link[1]))
                    edge_colors[i].append(colors[i])

        num_active_bonds = len(bond_type_indices)        
        pos = nx.spring_layout(G)
        offset_distance = 0.01
        
        plt.figure(figsize=(8, 6))
        
        for idx, i in enumerate(bond_type_indices):
            if len(edge_list[i]) > 0:
                for edge in edge_list[i]:
                    node1, node2 = edge
                    x1, y1 = pos[node1]
                    x2, y2 = pos[node2]
                    
                    # Calculate perpendicular offset
                    dx = x2 - x1
                    dy = y2 - y1
                    length = np.sqrt(dx**2 + dy**2)
                    
                    # Perpendicular unit vector
                    perp_x = -dy / length
                    perp_y = dx / length
                    
                    # Offset based on actual number of bond types present
                    offset = (idx - (num_active_bonds - 1) / 2) * offset_distance
                    
                    # New positions
                    new_x1 = x1 + perp_x * offset
                    new_y1 = y1 + perp_y * offset
                    new_x2 = x2 + perp_x * offset
                    new_y2 = y2 + perp_y * offset
                    
                    # Draw the offset line
                    plt.plot([new_x1, new_x2], [new_y1, new_y2], 
                            color=colors[i], linewidth=2, zorder=1)
        
        # draw nodes
        nx.draw_networkx_nodes(G, pos=pos, node_color='lightgreen', node_size=500, zorder=3)
        nx.draw_networkx_labels(G, pos=pos, font_size=16, font_weight='bold', zorder=4)
        
        legs = [Line2D([0], [0], color=colors[i], lw=2, label=all_bond_types[i]) 
                          for i in range(len(all_bond_types)) 
                          if all_bond_types[i] in bond_types]
        plt.legend(handles=legs, loc='best')
        plt.title(f'Graph of lattice links (max distance: {self.max_distance()}; total distance= {np.sum(self.get_distances())})')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return
    
    def max_distance(self):
        """
        Returns the maximal distance between two nodes of a link
        """
        max_dist = int(0)
        for bond in self._bonds:
            max_dist = max(max_dist, abs(bond.site1-bond.site2))
        return max_dist
    
    def get_distances(self):
        """
        Returns all distances bewteen nodes of links
        """
        distances = np.zeros(self._nbonds, dtype=int)
        for i, bond in enumerate(self._bonds):
            distances[i] = abs(bond.site1-bond.site2)
        distances = np.sort(distances)
        return distances
    
    @property
    def Ns(self):
        return self._Ns
    
    @property
    def nbonds(self):
        return self._nbonds
    
    @property
    def bonds(self):
        return self._bonds
    
    @property
    def links(self):
        return self._unique_links
    
    @property
    def indices(self):
        return self._indices
    
    @property
    def bond_orders(self):
        return self._bond_orders

class chainLattice(Lattice):
    """
    Chain lattice
    """
    
    def __init__(self, Ns, isPBC):
        """
        Constructor of a chain lattice with Ns sites, with nearest neighbor 
        isotropic bilinear Heisenberg interaction
        
        Parameters
        ----------
        Ns : int
            number of sites
        isPBC : bool
            True for periodic boundary conditions
            False for open boundary conditions
        
        Remarks
        -------
        - To add additional interactions, use add_bond(site1, site2, bond_type, coupling)
        - To modify the coupling value of a specific bond, use set_coupling(site1, site2, bond_type, coupling)
        """
        super().__init__()
        self._Ns = int(Ns)
        coupling = 1.0
        if isPBC==False:
            for i in range(0, self._Ns-1):
                self.add_bond(i, i+1, 'HB1', coupling)
        else:
            self.add_bond(0, 1, 'HB1', coupling)
            for i in range(0, self._Ns-1):
                self.add_bond(i, min(i+2, self._Ns-1), 'HB1', coupling)
        return


class Bond:
    """
    Class to represent an interaction bond
    """
    
    def __init__(self, site1, site2, bond_type, coupling):
        """
        Constructor of a bond
        
        Parameters
        ----------
        site1 : int
            first site in the bond
        site2 : int
            second site in the bond
        bond_type : str
            type of interaction 'HB', 'HB1', 'HB2', 'HB3', ...
        coupling : float
            interaction coupling
        
        Remarks
        -------
        For bilinear interaction, both 'HB' and 'HB1' are allowed as bond_type
        """
        assert(int(site1)!=int(site2))
        self._site1 = min(int(site1), int(site2))
        self._site2 = max(int(site1), int(site2))
        self.J = coupling
        if not bond_type[:2]=='HB':
            sys.exit('ERROR : Bond : __init__ : undefined bond_type.')
        if len(bond_type)==2:
            self._bond_type = 'HB1'
        else:
            self._bond_type = bond_type
        try:
            self._bond_order = int(self._bond_type[2:])
        except:
            sys.exit('ERROR : Bond : __init__ : undefined bond_type.')
        return
    
    def __str__(self):
        return f"Bond: ({self._site1}, {self._site2}), {self._bond_type}, J={self.J}"
    
    @property
    def bond(self):
        return np.array([self._site1, self._site2])
    
    @property
    def site1(self):
        return self._site1
    
    @property
    def site2(self):
        return self._site2

    @property
    def bond_type(self):
        return self._bond_type
    
    @property
    def bond_order(self):
        return self._bond_order
