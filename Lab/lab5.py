#1
a=[1,0,0,0,1,0,0,1]
m=[]
p=[]
k=3
g=0
for i in range (len(a)):
    if a[i]==1:
        m.append(i)
for j in range(len(m)-1):
    result=abs(m[j]-m[j+1])-1
    p.append(result)
for n in range(len(p)):
    if(p[n]==k):
        g=1
        break
if (g==1):
    print(True)
else:
    print(False)

print("\n")
#2
nums = [8, 2, 4, 7]
limit = 4
arr = []
ans = 0
j = 0
for i in range(len(nums)):
    arr.append(nums[i])
    arr.sort()    
    while arr[-1] - arr[0] > limit:
        arr.remove(nums[j])
        j += 1    
    ans = max(ans, i - j + 1)
print(ans)

print("\n")
#3
mat = [[1, 3, 11],[2, 4, 6]]
k = 5
ar = [0]
for i in mat:
    ar = sorted(a + b for a in ar for b in i)[:k]
output = ar[-1]
print(output)

print("\n")
#4
def countTriplets(arr): 
    n = len(arr) 
    count = 0 
    for i in range(n): 
        xor = 0 
        for j in range(i, n): 
            xor ^= arr[j] 
            if xor == 0: 
                count += j - i 
    return count 
arr1 = [2, 3, 1, 6, 7] 
print(countTriplets(arr1))

print("\n")
#5
def minTimeToCollectApples(n, edges, hasApple): 
    graph = [[] for _ in range(n)] 
    for u, v in edges: 
        graph[u].append(v) 
        graph[v].append(u) 
 
    def dfs(node, parent): 
        me = 0 
        for child in graph[node]: 
            if child != parent: 
                me += dfs(child, node) 
        if (hasApple[node] or node != 0) and me == 0: 
            return 2 
        return me 
 
    return max(0, dfs(0, -1) - 2) 
print(minTimeToCollectApples(7, [[0,1],[0,2],[1,4],[1,5],[2,3],[2,6]], 
[False,False,True,False,False,True,False])) 


print("\n")
#6
def waysToCutPizza(pizza, k): 
    rows, cols = len(pizza), len(pizza[0])
    MOD = 10**9 + 7
    dp = [[[0] * (k + 1) for _ in range(cols)] for _ in range(rows)] 
 
    for i in range(rows): 
        for j in range(cols): 
            dp[i][j][1] = 1 if 'A' in pizza[i][j:] else 0 
            for s in range(2, k + 1): 
                dp[i][j][s] = 0 
 
    for s in range(2, k + 1): 
        for i in range(rows - 1, -1, -1): 
            for j in range(cols - 1, -1, -1): 
                for x in range(i + 1, rows): 
                    if 'A' in pizza[i][j:]: 
                        dp[i][j][s] += dp[x][j][s - 1] 
                        dp[i][j][s] %= MOD 
                for y in range(j + 1, cols): 
                    if 'A' in [pizza[r][j] for r in range(i, rows)]: 
                        dp[i][j][s] += dp[i][y][s - 1] 
                        dp[i][j][s] %= MOD 
 
    return dp[0][0][k] 
print(waysToCutPizza(["A..","AA.","..."], 3))
