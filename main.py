import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt


class Graph(object):
    def draw(self):
        elist = list()
        for i in range(self.n):
            for j in range(self.n):
                if self.adjacency[i, j] == 1:
                    elist.append((i, j, self.weights[i, j]))

        G = nx.Graph()
        G.add_nodes_from(np.arange(self.n))
        G.add_weighted_edges_from(elist)
        labels_dictionary = dict()
        for i in range(self.n):
            labels_dictionary[i] = str(i) + '(' + str(self.peaks[i]) + ')'



        nx.draw_circular(G, labels=labels_dictionary)
        print(nx.cycle_basis(G, 0))
        #nx.draw(G, labels=labels_dictionary)


        plt.show()

    def __init__(self, a, b, c, d):

        def generate_peaks():
            for i in range(self.n):
                yield i

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
        self.adjacency = np.zeros((self.n, self.n), dtype=int)
        self.weights = np.zeros((self.n, self.n))

        for i in range(self.n):
            for j in range(self.n):
                calculated_value = math.sin((i * j * c + a) / d) + 1

                if 1 <= calculated_value <= 2:
                    self.adjacency[i, j] = 1
                    self.weights[i, j] = calculated_value * 10
                else:
                    self.weights[i, j] = float("Inf")

        self.ribs = []

        for i in range(self.n):
            self.ribs.append([])
            for j in range(self.n):
                if self.adjacency[i, j] == 1:
                    self.ribs[i].append(j)

        print(self.ribs)



        #print(self.weights)
        #for i in range(self.n - 1):
        #    self.ribs[2][i] = 0
        #    self.ribs[i][2] = 0

        #for i in range(self.n):
        #    self.ribs[2][i] = 0
        #    self.ribs[i][2] = 0




    def fsc(self):

        def Save(i, j, nZ, masQ, ftr):
            z = i
            while z != j:
                if z in masQ[nZ]:
                    break
                masQ[nZ].append(z)
                z = ftr[z]
            masQ[nZ].append(j)
            masQ[nZ].append(i)


        def Cicle(i, k, num, matrix_size, ftr, nz):
            k = k + 1
            num[i] = k
            for j in self.ribs[i]:
                if num[j] == 0:
                    ftr[j] = i
                    Cicle(j, k, num, matrix_size, ftr, nz)
                elif ftr[i] != j:
                    nz = nz + 1
                    Save(i, j, nz, masQ, ftr)


        n = 0
        m = 0

        num = []
        ftr = []

        matrix_size = len(self.ribs)

        for i in range(matrix_size):
            num.append(0)
            ftr.append(0)
            n = n + 1
            m = m + len(self.ribs[i])

        m = int(m / 2)

        masQ = []
        for i in range(m - n + 1):
            masQ.append([])


        k = 0
        nZ = 0

        Cicle(0, k, num, matrix_size, ftr, nZ)





        print("Фундаментальные циклы графа: ")
        ind = 0
        k = 0
        while k != len(masQ):
            if(len(masQ[k]) == 0):
                k = k + 1
                continue
            print("Цикл " + str(ind + 1) + " : ")
            print(masQ[k])
            print()
            ind = ind + 1
            k = k + 1

    def BellmanFord(self, src):

        # Step 1: Initialize distances from src to all other vertices
        # as INFINITE
        dist = [float("Inf")] * len(self.peaks)
        dist[src] = 0

        # Step 2: Relax all edges |V| - 1 times. A simple shortest
        # path from src to any other vertex can have at-most |V| - 1
        # edges
        for _ in range(len(self.peaks) - 1):
            # Update dist value and parent index of the adjacent vertices of
            # the picked vertex. Consider only those vertices which are still in
            # queue

            for i in range(len(self.weights)):
                for j in range(len(self.weights)):
                    if self.weights[i][j] != float("Inf") and dist[i] + self.weights[i][j] < dist[j]:
                        dist[j] = dist[i] + self.weights[i][j]

                    # Step 3: check for negative-weight cycles. The above step
        # guarantees shortest distances if graph doesn't contain
        # negative weight cycle. If we get a shorter path, then there
        # is a cycle.

        # print all distance
        print(dist)



    def prim(self):

        def isValidEdge(u, v, inMST):
            if u == v:
                return False
            if inMST[u] == False and inMST[v] == False:
                return False
            elif inMST[u] == True and inMST[v] == True:
                return False
            return True

        inMST = [False] * len(self.weights)

        # Include first vertex in MST
        inMST[0] = True

        # Keep adding edges while number of included
        # edges does not become V-1.
        edge_count = 0
        mincost = 0
        while edge_count < len(self.weights) - 1:

            # Find minimum weight valid edge.
            minn = float("inf")
            a = -1
            b = -1
            for i in range(len(self.weights)):
                for j in range(len(self.weights)):
                    if self.weights[i][j] < minn:
                        if isValidEdge(i, j, inMST):
                            minn = self.weights[i][j]
                            a = i
                            b = j

            if a != -1 and b != -1:
                print("Edge %d: (%d, %d) cost: %d" %
                      (edge_count, a, b, minn))
                edge_count += 1
                mincost += minn
                inMST[b] = inMST[a] = True

        print("Minimum cost = %d" % mincost)

    def dfs(self, start, visited=None, nestings = None, currentNesting = 1):
        if visited is None:
            visited = list()

        if nestings is None:
            nestings = list()
        nestings.append(currentNesting)

        visited.append(start)

        current = self.adjacency[start]
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
            current = self.adjacency[start]
            for next in set(np.where(current == 1)[0]) - set(visited):
                modified_dfs(next, visited)

            return frozenset(visited)

        for i in range(self.n):
            components.add(modified_dfs(i))

        return components




if __name__ == '__main__':
    G = Graph(1, 2, 3, 4)
    G.draw()
    print(G.dfs(2))
    print(G.сonnectivity_components())
    G.fsc()
    G.BellmanFord(3)
    G.prim()
