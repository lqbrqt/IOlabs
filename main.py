import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt


class Graph(object):
    def draw(self):
        elist = list()
        for i in range(self.n):
            for j in range(self.n):
                if self.ribs[i, j] == 1:
                    elist.append((i, j, self.weights[i, j]))

        G = nx.Graph()
        G.add_nodes_from(np.arange(self.n))
        G.add_weighted_edges_from(elist)
        labels_dictionary = dict()
        for i in range(self.n):
            labels_dictionary[i] = str(i) + '(' + str(self.peaks[i]) + ')'




        nx.draw_circular(G, labels=labels_dictionary)
        #nx.draw(G, labels=labels_dictionary)


        plt.show()

    def __init__(self, a, b, c, d):

        def generate_peaks():
            for i in range(self.n):
                yield b + i * c

        self.__a = a
        self.__b = b
        self.__c = c
        self.__d = d

        if a % 5 <= 1:
            self.n = 10
        elif a % 5 <= 3:
            self.n = 11
        else:
            self.n = 12

        self.peaks = np.array(list(generate_peaks()))
        self.ribs = np.zeros((self.n, self.n), dtype=int)
        self.weights = np.zeros((self.n, self.n))

        for i in range(self.n):
            for j in range(self.n):
                calculated_value = math.sin((i * j * c + a) / d) + 1

                if 1 <= calculated_value <= 2:
                    self.ribs[i, j] = 1
                    self.weights[i, j] = calculated_value * 10


        print(self.weights)
        #for i in range(self.n - 1):
        #    self.ribs[2][i] = 0
        #    self.ribs[i][2] = 0

        for i in range(self.n):
            self.ribs[2][i] = 0
            self.ribs[i][2] = 0



    def dfs(self, start, visited=None, nestings = None, currentNesting = 1):
        if visited is None:
            visited = list()

        if nestings is None:
            nestings = list()
        nestings.append(currentNesting)

        visited.append(start)

        current = self.ribs[start]
        for next in np.where(current == 1)[0]:
            if next in visited:
                pass
            else:
                self.dfs(next, visited, nestings, currentNesting + 1)

        return [visited, nestings]

    def сonnectivity_components(self):

        components = set()

        def modified_dfs(start, visited=None):
            if visited is None:
                visited = list()
            visited.append(start)
            current = self.ribs[start]
            for next in set(np.where(current == 1)[0]) - set(visited):
                modified_dfs(next, visited)

            return frozenset(visited)

        for i in range(self.n):
            components.add(modified_dfs(i))

        return components




if __name__ == '__main__':
    G = Graph(2, 4, 6, 8)
    G.draw()
    print(G.dfs(2))
    print(G.сonnectivity_components())
