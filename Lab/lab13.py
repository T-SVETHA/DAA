#1 word break
def wb(s,w):
    dp=[False]*(len(s)+1)
    dp[0]=True
    for i in range(1,len(s)+1):
        for j in range(i):
            if dp[j] and s[j:i] in w:
                dp[i]=True
                break
    return dp[len(s)]
s=input("string:")
w=input("give space").split()
print(wb(s,w))

#2 3-assembly line scheduling
def val(n):
    x = []
    for _ in range(n):
        num= int(input("Enter a number: "))
        x.append(num)
    return x        
def asl(a1, a2,a3, t1, t2, t3, e1, e2,e3, x1, x2, x3):
    n = len(a1)
    F1 = [0] * n
    F2 = [0] * n
    F3 = [0] * n
    F1[0] = e1 + a1[0]
    F2[0] = e2 + a2[0]
    F3[0]= e3+a3[0]
    for i in range(1, n):
        F1[i] = min(F1[i-1] + a1[i], F2[i-1] + t2[i-1] + a1[i])
        F2[i] = min(F2[i-1] + a2[i], F1[i-1] + t1[i-1] + a2[i])
        F3[i]= min (F3[i-1] + a3[i], F1[i-1] + t1[i-1] + a3[i], F2[i-1] + t2[i-1] + a3[i])
    f1 = F1[n-1] + x1
    f2 = F2[n-1] + x2
    f3 = F3[n-1] + x3 
    return min(f1, f2, f3)
a1 = int(input("How many numbers for a1? "))
r = val(a1)
print(r)
a2 = int(input("How many numbers for a2? "))
q = val(a2)
print(q)
a3 = int(input("How many numbers for a3? "))
b = val(a3)
print(b)
t1 = int(input("How many numbers for t1? "))
q1 = val(t1)
print(q1)
t2 = int(input("How many numbers for t2? "))
q2 = val(t2)
print(q2)
t3 = int(input("How many numbers for t3? "))
q3 = val(t3)
print(q3)
e1 = int(input("How many numbers for e1? "))
e2 = int(input("How many numbers for e2? "))
e3 = int(input("How many numbers for e3? "))
x1 = int(input("How many numbers for x1? "))
x2 = int(input("How many numbers for x2? "))
x3 = int(input("How many numbers for x3? "))
print(asl(r, q, b, q1, q2,q3, e1, e2, e3, x1, x2, x3))

#3 prims, kruskal.boruvka's
from collections import defaultdict
import heapq

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    def find_parent(self, parent, i):
        if parent[i] == i:
            return i
        return self.find_parent(parent, parent[i])

    def union(self, parent, rank, x, y):
        xroot = self.find_parent(parent, x)
        yroot = self.find_parent(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def kruskal(self):
        result = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = []
        rank = []
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
        while e < self.V - 1:
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find_parent(parent, u)
            y = self.find_parent(parent, v)
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
        return result

    def prim(self):
        result = []
        visited = [False] * self.V
        min_heap = [[0, 0]]  # (weight, vertex)
        while min_heap:
            weight, u = heapq.heappop(min_heap)
            if visited[u]:
                continue
            visited[u] = True
            result.append(u)
            for edge in self.graph:
                if edge[0] == u and not visited[edge[1]]:
                    heapq.heappush(min_heap, [edge[2], edge[1]])
                elif edge[1] == u and not visited[edge[0]]:
                    heapq.heappush(min_heap, [edge[2], edge[0]])
        return result

    def boruvka(self):
        parent = []
        rank = []
        cheapest = []
        numTrees = self.V
        MSTweight = 0
        MST = []

        for node in range(self.V):
            parent.append(node)
            rank.append(0)
            cheapest = [-1] * self.V

        while numTrees > 1:
            for i in range(len(self.graph)):
                u, v, w = self.graph[i]
                set1 = self.find_parent(parent, u)
                set2 = self.find_parent(parent, v)
                if set1 != set2:
                    if cheapest[set1] == -1 or cheapest[set1][2] > w:
                        cheapest[set1] = [u, v, w]
                    if cheapest[set2] == -1 or cheapest[set2][2] > w:
                        cheapest[set2] = [u, v, w]

            for node in range(self.V):
                if cheapest[node] != -1:
                    u, v, w = cheapest[node]
                    set1 = self.find_parent(parent, u)
                    set2 = self.find_parent(parent, v)
                    if set1 != set2:
                        MSTweight += w
                        MST.append([u, v, w])
                        self.union(parent, rank, set1, set2)
                        numTrees -= 1

            cheapest = [-1] * self.V

        return MST, MSTweight
g = Graph(4)
g.add_edge(0, 1, 10)
g.add_edge(0, 2, 6)
g.add_edge(0, 3, 5)
g.add_edge(1, 3, 15)
g.add_edge(2, 3, 4)

print("Kruskal's MST:", g.kruskal())
print("Prim's MST:", g.prim())
print("Boruvka's MST:", g.boruvka())

