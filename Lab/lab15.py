#1 sum of subsets
def subsets(nums, target):
    def backtrack(start, path, target):
        if target == 0:
            result.append(list(path))
            return
        if target < 0:
            return
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path, target - nums[i])
            path.pop()
    result = []
    backtrack(0, [], target)
    return result
nums = [5, 7, 6, 4, 8, 3, 2]
target = 15
subsets = subsets(nums, target)
for subset in subsets:
    print(subset)

#2 longest palindrome subsequence
def pali(s):
    n = len(s)
    if n == 0:
        return "", 0
    dp = [[False] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = True
    start = 0
    maxlen = 1
    for i in range(n-1):
        if s[i] == s[i+1]:
            dp[i][i+1] = True
            start = i
            maxlen = 2
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if dp[i+1][j-1] and s[i] == s[j]:
                dp[i][j] = True
                if length > maxlen:
                    start = i
                    maxlen = length
    substr= s[start:start + maxlen]
    return substr, maxlen
s = "theeh"
substring, length = pali(s)
print(f"{substring} Length of {length}")

#3 graph colouring
def gr(graph):
    numvertices = len(graph)
    result = [-1] * numvertices
    result[0] = 0
    available = [False] * numvertices   
    for u in range(1, numvertices):
        for i in graph[u]:
            if result[i] != -1:
                available[result[i]] = True       
        cr = 0
        while cr < numvertices:
            if not available[cr]:
                break
            cr += 1        
        result[u] = cr        
        for i in graph[u]:
            if result[i] != -1:
                available[result[i]] = False   
    for u in range(numvertices):
        print(f"Vertex {u} --->  Color {result[u]}")
graph = [
    [0,1],
    [1,2],
    [2,3],
    [3,0]
]
gr(graph)
