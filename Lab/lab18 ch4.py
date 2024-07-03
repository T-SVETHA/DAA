#1
def dice_throw(num_sides, num_dice, target):
    dp = [[0 for _ in range(target + 1)] for _ in range(num_dice + 1)]
    dp[0][0] = 1
    for i in range(1, num_dice + 1):
        for j in range(1, target + 1):
            dp[i][j] = 0
            for k in range(1, num_sides + 1):
                if j - k >= 0:
                    dp[i][j] += dp[i - 1][j - k]
    
    return dp[num_dice][target]

print("Number of ways to reach sum 7:", dice_throw(6, 2, 7))

#2
def assembly_line_scheduling(n, a1, a2, t1, t2, e1, e2, x1, x2):
    T1 = [0] * n
    T2 = [0] * n
    T1[0] = e1 + a1[0]
    T2[0] = e2 + a2[0]
    for i in range(1, n):
        T1[i] = min(T1[i-1] + a1[i], T2[i-1] + t2[i-1] + a1[i])
        T2[i] = min(T2[i-1] + a2[i], T1[i-1] + t1[i-1] + a2[i])
    return min(T1[n-1] + x1, T2[n-1] + x2)

n = 3
a1 = [4, 5, 3]
a2 = [2, 10, 1]
t1 = [0, 2, 4]
t2 = [7, 0, 9]
e1 = 10
e2 = 12
x1 = 18
x2 = 7
print("Minimum time required to process the product:", assembly_line_scheduling(n,a1,a2,t1,t2,e1,e2,x1,x2))

#3
def three_assembly_lines(num_stations, times, transfer_times, dependencies):
    n = num_stations
    L1, L2, L3 = times
    T = transfer_times
    dp = [[float('inf')] * n for _ in range(3)]
    dp[0][0], dp[1][0], dp[2][0] = L1[0], L2[0], L3[0]
    for i in range(1, n):
        dp[0][i] = min(dp[0][i-1] + L1[i], dp[1][i-1] + T[1][0] + L1[i], dp[2][i-1] + T[2][0] + L1[i])
        dp[1][i] = min(dp[1][i-1] + L2[i], dp[0][i-1] + T[0][1] + L2[i], dp[2][i-1] + T[2][1] + L2[i])
        dp[2][i] = min(dp[2][i-1] + L3[i], dp[0][i-1] + T[0][2] + L3[i], dp[1][i-1] + T[1][2] + L3[i])
    return min(dp[0][n-1], dp[1][n-1], dp[2][n-1])
num_stations = 3
times = [
    [5, 9, 3],
    [6, 8, 4],
    [7, 6, 5]
]
transfer_times = [
    [0, 2, 3],
    [2, 0, 4],
    [3, 4, 0]
]
dependencies = [(0, 1), (1, 2)]  
print("Minimum production time:", three_assembly_lines(num_stations,times,transfer_times,dependencies))

#4
from itertools import permutations
def tsp(matrix):
    n = len(matrix)
    all_sets = (1 << n) - 1
    dp = [[None] * n for _ in range(1 << n)]
    def tsp_util(mask, pos):
        if mask == all_sets:
            return matrix[pos][0]  
        if dp[mask][pos] is not None:
            return dp[mask][pos]
        min_cost = float('inf')
        for city in range(n):
            if mask & (1 << city) == 0:  
                new_cost = matrix[pos][city] + tsp_util(mask | (1 << city), city)
                min_cost = min(min_cost, new_cost)
        dp[mask][pos] = min_cost
        return dp[mask][pos]
    return tsp_util(1, 0)
matrix1 = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
print(tsp(matrix1))

#5
def longest_palindromic_substring(s: str) -> str:
    n = len(s)
    if n == 0:
        return ""
    start, max_length = 0, 1
    for i in range(1, n):
        low, high = i - 1, i
        while low >= 0 and high < n and s[low] == s[high]:
            if high - low + 1 > max_length:
                start = low
                max_length = high - low + 1
            low -= 1
            high += 1

        low, high = i - 1, i + 1
        while low >= 0 and high < n and s[low] == s[high]:
            if high - low + 1 > max_length:
                start = low
                max_length = high - low + 1
            low -= 1
            high += 1
    
    return s[start:start + max_length]
print(longest_palindromic_substring("babad"))

#6
def length_of_longest_substring(s: str) -> int:
    char_index = {}
    max_length = start = 0
    for i, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        char_index[char] = i
        max_length = max(max_length, i - start + 1)
    return max_length
print(length_of_longest_substring("abcabcbb"))  

#7
def word_break(s: str, wordDict) -> bool:
    word_set = set(wordDict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[-1]

print(word_break("leetcode", ["leet", "code"]))  

#8
def word_break_all(s: str, wordDict) -> bool:
    word_set = set(wordDict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    breaks = [[] for _ in range(len(s) + 1)]
    breaks[0] = [""]

    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                for prefix in breaks[j]:
                    if prefix:
                        breaks[i].append(prefix + " " + s[j:i])
                    else:
                        breaks[i].append(s[j:i])

    if dp[-1]:
        print("Yes")
        print(breaks[-1])
    else:
        print("No")

    return dp[-1]

word_break_all("ilike", ["i", "like", "sam", "sung", "samsung", "mobile", "ice", "cream", "icecream", "man", "go", "mango"])
word_break_all("ilikesamsung", ["i", "like", "sam", "sung", "samsung", "mobile", "ice", "cream", "icecream", "man", "go", "mango"])

#9
def full_justify(words, maxWidth):
    def add_spaces(i, space_slots, space, maxWidth):
        line = words[i[0]]
        for k in range(i[0] + 1, i[1]):
            if space_slots > 0:
                line += ' ' * (space // space_slots + 1)
                space -= space // space_slots + 1
                space_slots -= 1
            else:
                line += ' ' * (space // space_slots)
                space -= space // space_slots
            line += words[k]
        line += ' ' * space
        return line
    
    res = []
    i, n = 0, len(words)
    while i < n:
        count, length = 0, 0
        while i + count < n and length + len(words[i + count]) <= maxWidth - count:
            length += len(words[i + count])
            count += 1
        
        if i + count == n:
            res.append(' '.join(words[i:i+count]) + ' ' * (maxWidth - length - (count - 1)))
        else:
            res.append(add_spaces((i, i+count), count-1, maxWidth-length, maxWidth))        
        i += count
    return res
print(full_justify(["This", "is", "an", "example", "of", "text", "justification."], 16))

#10
class WordFilter:
    def _init_(self, words):
        self.prefix_suffix_map = {}
        for index, word in enumerate(words):
            length = len(word)
            for i in range(length + 1):
                for j in range(length + 1):
                    self.prefix_suffix_map[(word[:i], word[j:])] = index

    def f(self, pref, suff):
        return self.prefix_suffix_map.get((pref, suff), -1)
wordFilter = WordFilter(["apple"])
print(wordFilter.f("a","e"))

#11
def floyd_warshall(n, edges):
    inf = float('inf')
    dist = [[inf] * n for _ in range(n)]    
    for i in range(n):
        dist[i][i] = 0    
    for u, v, w in edges:
        dist[u][v] = w    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist
edges_a = [(0, 1, 3), (0, 2, 8), (0, 3, -4), (1, 3, 1), (1, 2, 4), (2, 0, 2), (3, 2, -5), (3, 1, 6)]
dist_matrix_a = floyd_warshall(4, edges_a)
print("Distance matrix (a):")
for row in dist_matrix_a:
    print(row)
print("Shortest path from City 1 to City 3 =", dist_matrix_a[0][2])  

#12
import sys
def floyd_warshall(graph):
    dist = list(map(lambda i: list(map(lambda j: j, i)), graph))
    V = len(graph)
    
    for k in range(V):
        for i in range(V):
            for j in range(V):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist
def print_solution(dist):
    V = len(dist)
    for i in range(V):
        for j in range(V):
            if dist[i][j] == sys.maxsize:
                print("INF", end=" ")
            else:
                print(dist[i][j], end=" ")
        print()
INF = sys.maxsize
graph = [
    [0, 3, INF, 7],
    [8, 0, 2, INF],
    [5, INF, 0, 1],
    [2, INF, INF, 0]
]
dist = floyd_warshall(graph)
print("Distance matrix before the link failure:")
print_solution(dist)
graph[1][3] = INF
dist_after_failure = floyd_warshall(graph)
print("\nDistance matrix after the link failure:")
print_solution(dist_after_failure)
print(f"\nShortest path from Router A to Router D before link failure: {dist[0][3]}")
print(f"Shortest path from Router A to Router D after link failure: {dist_after_failure[0][3]}")

#13
def floyd_warshall(graph):
    n = len(graph)
    dist = [[float('inf')] * n for _ in range(n)]
    for u in range(n):
        dist[u][u] = 0
    for u, v, w in graph:
        dist[u][v] = w
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist

def cities_within_distance(dist, threshold):
    count = 0
    for i in range(len(dist)):
        if sum(d <= threshold for d in dist[i]) - 1 > count:
            count = sum(d <= threshold for d in dist[i]) - 1
    return count

edges = [[0, 1, 2], [0, 4, 8], [1, 2, 3], [1, 4, 2], [2, 3, 1], [3, 4, 1]]
n = 5
threshold = 2
dist = floyd_warshall(edges)
result = cities_within_distance(dist, threshold)
print(result)

#14
def optimal_bst(keys, freq):
    n = len(keys)
    cost = [[0 for _ in range(n)] for _ in range(n)]
    root = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        cost[i][i] = freq[i]
        root[i][i] = i
    
    for L in range(2, n + 1):
        for i in range(n - L + 1):
            j = i + L - 1
            cost[i][j] = float('inf')
            for r in range(i, j + 1):
                c = ((cost[i][r - 1] if r > i else 0) +
                     (cost[r + 1][j] if r < j else 0) +
                     sum(freq[i:j + 1]))
                if c < cost[i][j]:
                    cost[i][j] = c
                    root[i][j] = r
    
    return cost, root

keys = ['A', 'B', 'C', 'D']
freq = [0.1, 0.2, 0.4, 0.3]

cost, root = optimal_bst(keys, freq)

print("Cost Table:")
for row in cost:
    print(row)

print("\nRoot Table:")
for row in root:
    print(row)

#15
def optimal_bst(keys, freq, n):
    cost = [[0 for x in range(n)] for y in range(n)]
    root = [[0 for x in range(n)] for y in range(n)]

    for i in range(n):
        cost[i][i] = freq[i]
        root[i][i] = i

    for L in range(2, n+1):
        for i in range(n-L+1):
            j = i + L - 1
            cost[i][j] = float('inf')

            for r in range(i, j+1):
                c = (cost[i][r-1] if r > i else 0) + (cost[r+1][j] if r < j else 0) + sum(freq[i:j+1])
                if c < cost[i][j]:
                    cost[i][j] = c
                    root[i][j] = r

    return cost, root
keys = [10, 12, 16, 21]
freq = [4, 2, 6, 3]
n = len(keys)
cost, root = optimal_bst(keys, freq, n)
print("Cost Matrix:")
for row in cost:
    print(row)
print("\nRoot Matrix:")
for row in root:
    print(row)

print(f"\nOptimal Cost: {cost[0][n-1]}")

#16
def catMouseGame(graph):
    DRAW, MOUSE, CAT = 0, 1, 2
    N = len(graph)
    color = [[[DRAW] * N for _ in range(N)] for _ in range(2)]
    degree = [[[0] * N for _ in range(N)] for _ in range(2)]
    
    for m in range(N):
        for c in range(N):
            degree[1][m][c] = len(graph[m])
            degree[0][m][c] = len(graph[c]) - (0 in graph[c])

    queue = []

    for i in range(N):
        for t in range(2):
            color[t][0][i] = MOUSE
            queue.append((0, i, t, MOUSE))
            if i > 0:
                color[t][i][i] = CAT
                queue.append((i, i, t, CAT))

    while queue:
        i, j, t, c = queue.pop(0)
        for ni, nj, nt in parents(graph, i, j, t):
            if color[nt][ni][nj] == DRAW:
                if nt == 1 and c == MOUSE or nt == 0 and c == CAT:
                    color[nt][ni][nj] = c
                    queue.append((ni, nj, nt, c))
                else:
                    degree[nt][ni][nj] -= 1
                    if degree[nt][ni][nj] == 0:
                        color[nt][ni][nj] = c
                        queue.append((ni, nj, nt, c))
                        
    return color[1][1][2]

def parents(graph, i, j, t):
    if t == 1:
        for ni in graph[i]:
            yield (ni, j, 1-t)
    else:
        for nj in graph[j]:
            if nj > 0:
                yield (i, nj, 1-t)

graph1 = [[2,5],[3],[0,4,5],[1,4,5],[2,3],[0,2,3]]
print(catMouseGame(graph1)) # Output: 0'''

#17
import heapq
def maxProbability(n, edges, succProb, start, end):
    graph = [[] for _ in range(n)]
    for (a, b), prob in zip(edges, succProb):
        graph[a].append((b, prob))
        graph[b].append((a, prob))

    pq = [(-1, start)]
    probabilities = [0] * n
    probabilities[start] = 1
    while pq:
        prob, node = heapq.heappop(pq)
        prob = -prob

        if node == end:
            return prob

        for neighbor, edge_prob in graph[node]:
            new_prob = prob * edge_prob
            if new_prob > probabilities[neighbor]:
                probabilities[neighbor] = new_prob
                heapq.heappush(pq, (-new_prob, neighbor))

    return 0.0

n1 = 3
edges1 = [[0,1],[1,2],[0,2]]
succProb1 = [0.5, 0.5, 0.2]
start1 = 0
end1 = 2
print(f"{maxProbability(n1, edges1, succProb1, start1, end1):.5f}")

#18
def uniquePaths(m, n):
    dp = [[1] * n for _ in range(m)]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    
    return dp[m-1][n-1]


m1, n1 = 3, 7
print(uniquePaths(m1, n1))

#19
def numIdenticalPairs(nums):
    count = {}
    good_pairs = 0
    
    for num in nums:
        if num in count:
            good_pairs += count[num]
            count[num] += 1
        else:
            count[num] = 1
    
    return good_pairs

nums1 = [1,2,3,1,1,3]
print(numIdenticalPairs(nums1))

#20
def findTheCity(n, edges, distanceThreshold):
    import heapq

    graph = {i: [] for i in range(n)}
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    def dijkstra(start):
        distances = [float('inf')] * n
        distances[start] = 0
        heap = [(0, start)]

        while heap:
            curr_dist, node = heapq.heappop(heap)
            if curr_dist > distances[node]:
                continue

            for neighbor, weight in graph[node]:
                distance = curr_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(heap, (distance, neighbor))
        
        return distances

    min_reachable_cities = float('inf')
    result_city = -1

    for i in range(n):
        reachable_cities = sum(1 for dist in dijkstra(i) if dist <= distanceThreshold)
        if reachable_cities <= min_reachable_cities:
            min_reachable_cities = reachable_cities
            result_city = i
    
    return result_city

n1 = 4
edges1 = [[0,1,3],[1,2,1],[1,3,4],[2,3,1]]
distanceThreshold1 = 4
print(findTheCity(n1, edges1, distanceThreshold1))

#21
import heapq
def networkDelayTime(times, n, k):
    graph = {i: [] for i in range(1, n+1)}
    for u, v, w in times:
        graph[u].append((v, w))
    heap = [(0, k)]
    distances = {i: float('inf') for i in range(1, n+1)}
    distances[k] = 0
    while heap:
        time, node = heapq.heappop(heap)
        if time > distances[node]:
            continue
        for neighbor, weight in graph[node]:
            d = time + weight
            if d < distances[neighbor]:
                distances[neighbor] = d
                heapq.heappush(heap, (d, neighbor))
    max_distance = max(distances.values())
    return max_distance if max_distance < float('inf') else -1
times1 = [[2,1,1],[2,3,1],[3,4,1]]
n1, k1 = 4, 2
print(networkDelayTime(times1, n1, k1))  
