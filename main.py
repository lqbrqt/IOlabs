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
                    elist.append((i, j, int(self.weights[i, j])))

        vertexLabels = dict()

        for i in range(self.n):
            vertexLabels[i] = str(i)


        G = nx.Graph()
        G.add_nodes_from(np.arange(self.n))
        pos = nx.spring_layout(G)
        G.add_weighted_edges_from(elist)
        labels_dictionary = dict()
        for i in range(len(self.weights)):
            for j in range(len(self.weights)):
                if(self.weights[i, j] != float("Inf")):
                    labels_dictionary[i, j] = int(self.weights[i, j])
        nx.draw_networkx_edge_labels(G,pos, edge_labels=labels_dictionary)




        nx.draw_circular(G, labels = vertexLabels)
        #print(nx.cycle_basis(G, 0))
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

        h = 1

        for i in range(self.n):
            for j in range(self.n):
                calculated_value = math.sin((i * j * c + a) / d) + 1


                if 1 <= calculated_value <= 2:
                    self.adjacency[i, j] = 1
                    self.weights[i, j] = calculated_value * 10

                else:
                    self.weights[i, j] = float("Inf")

                if (h % 2 == 0 and h % 3 == 0 and h % 4 == 0):
                    self.weights[i, j] = -100

                if(i == j):
                    self.weights[i, j] = 0
                    self.adjacency[i, j] = 0
                h = h + 1

        self.ribs = []

        for i in range(self.n):
            self.ribs.append([])
            for j in range(self.n):
                if self.adjacency[i, j] == 1:
                    self.ribs[i].append(j)

        #print(self.ribs)



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





        print("?????????????????????????????? ?????????? ??????????: ")
        ind = 0
        k = 0
        while k != len(masQ):
            if(len(masQ[k]) == 0):
                k = k + 1
                continue
            print("???????? " + str(ind + 1) + " : ")
            print(masQ[k])
            print()
            ind = ind + 1
            k = k + 1

    def BellmanFord(self, src):

        dist = [float("Inf")] * len(self.peaks)
        predecessor = [src] * len(self.peaks)

        dist[src] = 0

        for _ in range(len(self.peaks) - 1):

            for i in range(len(self.weights)):
                for j in range(len(self.weights)):
                    if self.weights[i][j] != float("Inf") and dist[i] + self.weights[i][j] < dist[j]:
                        predecessor[j] = i
                        dist[j] = dist[i] + self.weights[i][j]

        neg = []

        for i in range(len(self.weights)):
            for j in range(len(self.weights)):
                if dist[i] != float("Inf") and dist[i] + self.weights[i][j] < dist[j] :
                    neg.append(i)
                    print("?? ?????????? ???????????????????? ?????????? ???????????????????????????? ????????. ?????? " + str(i) + " ??????????????")

        globalPath = []

        for i in range(len(self.weights)):
            path = []
            current = i
            while current != src:
                if current in neg:
                    path = [-float("inf")]
                    break
                path.append(current)
                current = predecessor[current]

            globalPath.append(path[::-1])

        print(globalPath)



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

        inMST[0] = True


        edge_count = 0
        mincost = 0
        while edge_count < len(self.weights) - 1:

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

    def ??onnectivity_components(self):

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
    G = Graph(14, 15, 16, 17)
    G.draw()
    #print(G.dfs(2))
    #print(G.??onnectivity_components())
    #G.fsc()
    G.BellmanFord(6)
    #G.prim()
