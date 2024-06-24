#1. Convert the Temperature 
 
def convert(celsius): 
    fahrenheit = celsius * 9/5 + 32 
    kelvin = celsius + 273.15 
    return fahrenheit, kelvin 
celsius = 25 
fahrenheit, kelvin = convert(celsius) 
print(fahrenheit, kelvin) 
 
 
#2. Number of Subarrays With LCM Equal to K 
 
from math import gcd 
from functools import reduce 
 
def lcm(a, b): 
    return a * b // gcd(a, b) 
 
def number(arr, k): 
    def subarray(array): 
        return reduce(lcm, array) 
 
    count = 0 
    for i in range(len(arr)): 
        for j in range(i + 1, len(arr) + 1): 
            if subarray(arr[i:j]) == k: 
                count += 1 
    return count 
arr = [2, 3, 4, 6] 
k = 12 
result = number(arr, k) 
print(result) 
 
 
#3. Minimum Number of OperaTIons to Sort a Binary Tree by Level 
 
from collections import deque 
class TreeNode: 
    def __init__(self, val=0, le =None, right=None): 
        self.val = val 
        self.le = le 
        self.right = right 
 
def min(root): 
    if not root: 
        return 0 
 
    queue = deque([root]) 
    opera ons = 0 
 
    while queue: 
        levelsize = len(queue) 
        currentlevel = [] 
 
        for _ in range(levelsize): 
            node = queue.pople () 
            currentlevel.append(node.val) 
            if node.le : 
                queue.append(node.le ) 
            if node.right: 
                queue.append(node.right) 
 
        sortedlevel = sorted(currentlevel) 
        operations += sum(1 for i in range(len(currentlevel)) if currentlevel[i] != 
sortedlevel[i]) 
 
    return operations 
root = TreeNode(1, TreeNode(3, TreeNode(5), TreeNode(7)), TreeNode(2, TreeNode(6), 
TreeNode(4))) 
result = min(root) 
print(result) 
 
 
#4. Maximum Number of Non-overlapping Palindrome Substrings 
 
def max(s): 
    n = len(s) 
    dp = [[False] * n for _ in range(n)] 
 
    for i in range(n): 
        dp[i][i] = True 
 
    for length in range(2, n + 1): 
        for i in range(n - length + 1): 
            j = i + length - 1 
            if s[i] == s[j]: 
                if length == 2 or dp[i + 1][j - 1]: 
                    dp[i][j] = True 
 
    count = 0 
    end = -1 
 
    for i in range(n): 
        if dp[end + 1][i]: 
            count += 1 
            end = i 
 
    return count 
 
s = "ababa" 
result = max(s) 
print(result) 
 
 
#5. Minimum Cost to Buy Apples 
def min(cost, quantity, k): 
    n = len(cost) 
    dp = [[float('inf')] * (k + 1) for _ in range(n + 1)] 
    dp[0][0] = 0 
 
    for i in range(1, n + 1): 
        for j in range(k + 1): 
            dp[i][j] = dp[i - 1][j] 
            if j >= quantity[i - 1]: 
                dp[i][j] = min(dp[i][j], dp[i - 1][j - quantity[i - 1]] + cost[i - 1]) 
 
    return dp[n][k] if dp[n][k] != float('inf') else -1 
 
cost = [2, 3, 5] 
quantity = [1, 2, 3] 
k = 5 
result = min(cost, quantity, k) 
print(result) 
 
 
#6. Customers With Strictly Increasing Purchases 
 
#7. Number of Unequal Triplets in Array 
 
 
  def triplets(arr): 
    n = len(arr) 
    count = 0 
 
    for i in range(n): 
        for j in range(i + 1, n): 
            for k in range(j + 1, n): 
                if arr[i] != arr[j] and arr[j] != arr[k] and arr[i] != arr[k]: 
                    count += 1 
    return count 
arr = [1, 2, 3, 4] 
result = triplets(arr) 
print(result) 
 
 
#8. Closest Nodes Queries in a Binary Search Tree 
 
class TreeNode: 
    def __init__(self, val=0, le =None, right=None): 
        self.val = val 
        self.le = le 
        self.right = right 
 
def closest_nodes(root, queries): 
    def inorder_traversal(node): 
        return inorder_traversal(node.le ) + [node.val] + inorder_traversal(node.right) if node 
else [] 
 
    sorted_vals = inorder_traversal(root) 
    result = [] 
 
    for q in queries: 
        pos = bisect.bisect_le (sorted_vals, q) 
        if pos == 0: 
            result.append(sorted_vals[0]) 
        elif pos == len(sorted_vals): 
            result.append(sorted_vals[-1]) 
        else: 
            if abs(sorted_vals[pos] - q) < abs(sorted_vals[pos - 1] - q): 
                result.append(sorted_vals[pos]) 
            else: 
                result.append(sorted_vals[pos - 1]) 
 
    return result 
root = TreeNode(4, TreeNode(2, TreeNode(1), TreeNode(3)), TreeNode(6, TreeNode(5), 
TreeNode(7))) 
queries = [3, 8] 
result = closest_nodes(root, queries) 
print(result) 
 
 
#9. Minimum Fuel Cost to Report to the Capital  
def min(n, edges, price): 
    from collections import defaultdict, deque 
 
    graph = defaultdict(list) 
    for u, v in edges: 
        graph[u].append(v) 
        graph[v].append(u) 
 
    def bfs(start): 
        queue = deque([start]) 
        visited = set([start]) 
        fuelcost = 0 
 
        while queue: 
            node = queue.pople () 
            for neighbor in graph[node]: 
                if neighbor not in visited: 
                    visited.add(neighbor) 
                    queue.append(neighbor) 
                    fuelcost += price 
        return fuelcost 
 
    return bfs(0) 
n = 5 
edges = [(0, 1), (1, 2), (1, 3), (3, 4)] 
price = 2 
result = min(n, edges, price) 
print(result) 
  
 
#10. Number of Beautiful Partions 
 
def number(s, k): 
    n = len(s) 
    dp = [[0] * (k + 1) for _ in range(n + 1)] 
    dp[0][0] = 1 
 
    for i in range(1, n + 1): 
        for j in range(1, k + 1): 
            for l in range(i): 
                if s[l:i] == s[l:i][::-1]: 
                    dp[i][j] += dp[l][j - 1] 
 
    return dp[n][k] 
s = "aab" 
k = 2 
result = number(s, k) 
print(result)       
