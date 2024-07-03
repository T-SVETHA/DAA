#1 max and min
def mm(arr, low, high): 
  if low == high: 
    return arr[low], arr[low] 
  mid = (low + high) // 2 
  lmin, lmax = mm(arr, low, mid) 
  rmin, rmax = mm(arr, mid + 1, high) 
  return min(lmin, rmin), max(lmax, rmax) 
arr = [10, 2, 6, 7, 4, 1, 9] 
min, max= mm(arr, 0, len(arr) - 1) 
print("Minimum element:", min) 
print("Maximum element:", max)

#2 max min of sorted array
def mm(arr, low, high): 
  if low == high: 
    return arr[low], arr[low] 
  mid = (low + high) // 2 
  lmin, lmax = mm(arr, low, mid) 
  rmin, rmax = mm(arr, mid + 1, high) 
  return min(lmin, rmin), max(lmax, rmax) 
arr = [1,2,3,4,5,6,7,8,9] 
min, max= mm(arr, 0, len(arr) - 1) 
print("Minimum element:", min) 
print("Maximum element:", max)

#3 merge sort
def merge(arr):
    k=0
    if len(arr) > 1: 
        mid = len(arr) // 2 
        L = arr[:mid] 
        R = arr[mid:]  
        merge(L) 
        merge(R)  
        i = j = k = 0 
        while i < len(L) and j < len(R):
            if L[i] < R[j]: 
                arr[k] = L[i] 
                i += 1 
            else: 
                arr[k] = R[j] 
                j += 1 
            k += 1
        while i < len(L): 
            arr[k] = L[i] 
            i += 1 
            k += 1
        while j < len(R): 
            arr[k] = R[j] 
            j += 1 
            k += 1

    return k,arr
arr = [12, 11, 13, 5, 6, 7] 
c,arr=merge(arr) 
print(f"count:{c},Sorted array is:", arr)

#4 merge sort
def merge(arr):
    k=0
    if len(arr) > 1: 
        mid = len(arr) // 2 
        L = arr[:mid] 
        R = arr[mid:]  
        merge(L) 
        merge(R)  
        i = j = k = 0 
        while i < len(L) and j < len(R):
            if L[i] < R[j]: 
                arr[k] = L[i] 
                i += 1 
            else: 
                arr[k] = R[j] 
                j += 1 
            k += 1
        while i < len(L): 
            arr[k] = L[i] 
            i += 1 
            k += 1
        while j < len(R): 
            arr[k] = R[j] 
            j += 1 
            k += 1

    return k,arr
arr = [12, 11, 13, 5, 6, 7] 
c,arr=merge(arr) 
print(f"count:{c},Sorted array is:", arr)

#5 quick sort
def quick(arr): 
    if len(arr) <= 1: 
        return arr  
    p = arr[len(arr) // 2] 
    l = [x for x in arr if x < p] 
    m = [x for x in arr if x == p] 
    r = [x for x in arr if x > p]  
    return quick(l) + m + quick(r) 
arr = [3, 6, 8, 10, 1, 2, 1] 
s = quick(arr) 
print("Sorted array:", s)

#6 quick sort
def quick(arr): 
    if len(arr) <= 1: 
        return arr  
    p = arr[len(arr) // 2] 
    l = [x for x in arr if x < p] 
    m = [x for x in arr if x == p] 
    r = [x for x in arr if x > p]  
    return quick(l) + m + quick(r) 
arr = [3, 6, 8, 10, 1, 2, 1] 
s = quick(arr) 
print("Sorted array:", s)

#9 k closest pair
import math

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def brute_force(points):
    min_dist = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d = distance(points[i], points[j])
            if d < min_dist:
                min_dist = d
    return min_dist

def closest_pair_strip(strip, delta):
    min_dist = delta
    strip.sort(key=lambda p: p[1])
    n = len(strip)
    for i in range(n):
        for j in range(i + 1, n):
            if (strip[j][1] - strip[i][1]) < min_dist:
                d = distance(strip[i], strip[j])
                if d < min_dist:
                    min_dist = d
            else:
                break
    return min_dist

def closest_pair_recursive(points):
    n = len(points)
    if n <= 3:
        return brute_force(points)
    
    mid = n // 2
    mid_point = points[mid]

    dl = closest_pair_recursive(points[:mid])
    dr = closest_pair_recursive(points[mid:])
    delta = min(dl, dr)
    
    strip = [p for p in points if abs(p[0] - mid_point[0]) < delta]
    
    return min(delta, closest_pair_strip(strip, delta))

def closest_pair(points):
    points.sort(key=lambda p: p[0])
    return closest_pair_recursive(points)

points = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
result = closest_pair(points)
print("The smallest distance is", result)

#10
def count(A, B, C, D):
    from collections import defaultdict
    sum_ab = defaultdict(int)
    for a in A:
        for b in B:
            sum_ab[a + b] += 1
    count = 0
    for c in C:
        for d in D:
            count += sum_ab[-(c + d)]
    return count
A = [1, 2]
B = [-2, -1]
C = [-1, 2]
D = [0, 2]
print("Output:", count(A, B, C, D))
