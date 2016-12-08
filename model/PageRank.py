#!/usr/bin/env python

# Given a graph (expressed using networkx package), apply PageRank algorithm to it and compute the importance of
# its nodes according to their linkages. For the details of PageRank algorithm, please check the corresponding
# webpage on Wikipedia.

import os
import sys
import csv
import operator
import numpy as np
from networkx.exception import NetworkXError
import matplotlib.pyplot as plt
import networkx as nx
sys.path.append("../util/")
import dataloader as dl

class PageRank:
    def __init__(self, graph, directed):
        self.graph = graph
        self.V = len(self.graph.nodes()) # number of nodes
        self.E = len(self.graph.edges()) # number of edges
        self.directed = directed
        self.ranks = dict() # page ranks

    def compute_ranks(self, damp=0.85, max_iter=100, tol=1e-6, personalization=None):
        # initialize rankings
        if personalization is None:
            for node, node_attr in self.graph.nodes(data=True):
                if self.directed:
                    self.ranks[node] = 1/float(self.V)
                else:
                    self.ranks[node] = node_attr.get("rank")
        else:
            missing = set(self.graph) - set(personalization)
            if missing:
                raise NetworkXError("Missing values for %s nodes" % missing)
            val_sum = float(sum(personalization.values()))
            self.ranks = dict((k, v/val_sum) for k, v in personalization.items())
        # update pagerank values over iterations
        count_iter = 0
        error = np.inf
        prev_rank = dict(self.ranks)
        while (count_iter<max_iter and error>tol):
            error = 0.0
            count_iter += 1
            for node in self.graph.nodes():
                rank_sum = 0
                if self.directed:
                    for edge in self.graph.in_edges(node):
                        nd_src = edge[0]
                        out_degree = self.graph.out_degree(nd_src)
                        if out_degree > 0:
                            rank_sum += self.ranks[nd_src] / float(out_degree)
                else:
                    for nd_src in self.graph.neighbors(node):
                        out_degree = self.graph.degree(nd_src)
                        if out_degree > 0:
                            rank_sum += self.ranks[nd_src] / float(out_degree)
                self.ranks[node] = damp*rank_sum + (1-damp)/float(self.V)
                error += (self.ranks[node] - prev_rank[node])**2
            error = np.sqrt(error / float(self.V))
            prev_rank = dict(self.ranks)

    def get_ranks(self):
        return self.ranks

    # check if the node is in the graph
    def check_node(self, nodeId):
        return (nodeId in self.graph.nodes())

    # get the top-k ranked neighbor given a node
    def get_top_influential_neighbors(self, nodeId, topk=10):
        if self.check_node(nodeId) == False:
            print "Cannot find the node in the graph!"
            return None
        neighbors = self.graph.neighbors(nodeId)
        neighbor_ranks = dict(zip(neighbors, operator.itemgetter(*neighbors)(self.ranks)))
        neighbor_ranks = sorted(neighbor_ranks.iteritems(),
                                key=operator.itemgetter(1), reverse=True) # sort neighbors by rank, return list of tuples
        neighbor_ranks = neighbor_ranks[:topk] # select top ranked neighbors
        return neighbor_ranks

    # draw a subgraph to present node and its top neighbors
    def draw_top_neighbor_graph(self, nodeId, topk=10):
        neighbor_rank_list = self.get_top_influential_neighbors(nodeId, topk)
        if neighbor_rank_list is None:
            print "No neighbor found!"
        sg = nx.Graph()
        node_size_scale = 1000 / self.ranks[nodeId]
        nodes = [nodeId]
        ranks = [self.ranks[nodeId]*node_size_scale]
        node_labels = {nodeId: nodeId}
        for nd, rk in neighbor_rank_list:
            nodes.append(nd)
            ranks.append(rk*node_size_scale)
            node_labels[nd] = nd
        sg.add_star(nodes=nodes, weights=ranks)
        plt.figure(1)
        pos = nx.spring_layout(sg) # set position of nodes
        nx.draw(sg, pos, labels=node_labels, node_color=ranks, node_size=ranks, cmap=plt.cm.Blues)
        savepath = '../result/star_graph_node'+str(nodeId)+'.png'
        plt.savefig(savepath)
        plt.show()

    # draw the whole
    def draw_graph(self):
        plt.figure(0)
        #pos = nx.spring_layout(self.graph, iterations=200)
        nx.draw(self.graph, node_color='red', node_size=10, width=1.0)
        #nx.draw_networkx_nodes(self.graph, node_color='red', node_size=100)
        #nx.draw_networkx_edges(self.graph,  width=1.0)
        plt.savefig('../result/all_user_graph.png')
        plt.show()


# helper function to save the ranking result into ../result folder
def save_results(ranks, filename):
    save_path = os.path.join('../result', filename)
    with open(save_path, 'wb') as fout:
        cwriter = csv.writer(fout, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for rk in ranks:
            cwriter.writerow([str(rk[0]), str(rk[1])])
    fout.close()

# main function for module test
def main():
    if len(sys.argv)==1:
        print("Usage: python PageRank.py <data_filename> <directed OR undirected>")
        sys.exit(1)
    data_filename = sys.argv[1]
    isDirected = True if sys.argv[2]=="directed" else False
    graph = dl.read_network_data(data_filename, isDirected)
    pagerank = PageRank(graph, isDirected)
    pagerank.compute_ranks(0.85)
    ranks_of_page = pagerank.get_ranks()
    sorted_ranks = sorted(ranks_of_page.iteritems(), key=operator.itemgetter(1), reverse=True)
    save_filename = data_filename.split('/')[-1].split('.')[0] + ".csv"
    save_results(sorted_ranks, save_filename)
    pagerank.draw_top_neighbor_graph('12', 50)
    pagerank.draw_graph()


if __name__ == "__main__":
    main()