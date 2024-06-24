#1.Height of Binary Tree A er Subtree Removal 
class TreeNode: 
    def __init__(self, val=0, le =None, right=None): 
        self.val = val 
        self.le = le 
        self.right = right  
def height(root): 
    if not root: 
        return 0 
    return 1 + max(height(root.le ), height(root.right))  
def remove(root, target): 
    if not root: 
        return None, 0 
    if root.val == target: 
        return None, 0 
    root.le , le height = remove(root.le , target) 
    root.right, rightheight = remove(root.right, target) 
    return root, height(root) 
root = TreeNode(1) 
root.le = TreeNode(2) 
root.right = TreeNode(3) 
root.le .le = TreeNode(4) 
root.le .right = TreeNode(5) 
neWroot, newheight = remove(root, 2) 
print("New height of the tree:", newheight) 
 
 
#2. Sort Array by Moving Items 
def sortarray(arr): 
    arr.sort() 
    return arr 
arr = [5, 3, 1, 4, 2] 
sorted_arr = sort_array(arr) 
print("Sorted array:", sorted_arr) 

#3. Apply Operations 
def apply(arr): 
    result = [] 
    for num in arr: 
        result.append(num * num) 
    return result 
arr = [1, 2, 3, 4] 
result = apply(arr) 
print("Result array:", result) 

#4. Maximum Sum of Distinct Subarrays With Length K 
def max(nums, k): 
    maxsum = 0 
    n = len(nums) 
   for i in range(n - k + 1): 
        subarray = nums[i:i + k] 
        if len(set(subarray)) == k: 
            maxsum = max(maxsum, sum(subarray))     
    return maxsum 
nums = [1, 2, 1, 3, 4] 
k = 3 
print("Maximum sum of dis nct subarrays of length", k, ":", max(nums,k)) 

#5. Total Cost to Hire K Workers 
def total(costs, k): 
    costs.sort() 
    return sum(costs[:k]) 
costs = [10, 20, 30, 40, 50] 
k = 3 
print("Total cost to hire", k, "workers:", total(costs, k)) 

#6. Minimum Total Distance Traveled 
def min(points): 
    points.sort() 
    median = points[len(points) // 2] 
    return sum(abs(point - median) for point in points) 
points = [1, 2, 3, 4, 5] 
print("Minimum total distance traveled:", min(points)) 
 
#7. Minimum Subarrays in a Valid Split 
def min(arr, max_sum): 
    subarraysum = 0 
    count = 1 
    for num in arr: 
        if subarraysum + num > maxsum: 
            count += 1 
            subarraysum = num 
        else: 
            subarraysum += num 
    return count 
arr = [1, 2, 3, 4, 5] 
maxsum = 5 
print("Minimum subarrays to split:", min(arr, maxsum)) 
 
#8. Number of Distinct Averages 
def dis(arr): 
    avgs = set() 
    for i in range(len(arr)): 
        for j in range(i + 1, len(arr)): 
            avg = (arr[i] + arr[j]) / 2 
            avgs.add(avg) 
    return len(avgs) 
arr = [1, 2, 3, 4] 
print("Number of dis nct averages:", dis (arr)) 
 
#9. Count Ways To Build Good Strings 
def count(s): 
    def isgood(s): 
        return s == s[::-1] 
     
    n = len(s) 
    count = 0 
    for i in range(n): 
        for j in range(i + 1, n + 1): 
            if isgood(s[i:j]): 
                count += 1 
    return count 
s = "aba" 
print("Number of ways to build good strings:", count(s)) 

#10. Most Profitable Path in a Tree 
class TreeNode: 
    def __init__(self, val=0, le =None, right=None): 
        self.val = val 
        self.le = le 
        self.right = right 
 
def max(root): 
    def dfs(node): 
        if not node: 
            return 0, 0  
        le profit, le sum = dfs(node.le ) 
        rightprofit, rightsum = dfs(node.right) 
        pathsum = node.val + max(le sum, rightsum) 
        maxprofit = max(le profit, rightprofit, pathsum) 
        return maxprofit, pathsum 
 
    maxprofit  = dfs(root) 
    return maxprofit 
root = TreeNode(5) 
root.le = TreeNode(4) 
root.right = TreeNode(8) 
root.le .le = TreeNode(11) 
root.le .le .le = TreeNode(7) 
root.le .le .right = TreeNode(2) 
root.right.le = TreeNode(13) 
root.right.right = TreeNode(4) 
root.right.right.right = TreeNode(1) 
print("Most profitable path in the tree:", max(root)) 
