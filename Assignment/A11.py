#1 median of medians
def median_of_medians(arr, k): 
    if len(arr) <= 5: 
        return sorted(arr)[k-1]      
    sublists = [arr[i:i + 5] for i in range(0, len(arr), 5)] 
    medians = [sorted(sublist)[len(sublist) // 2] for sublist in sublists] 
    pivot = median_of_medians(medians, len(medians) // 2)  
    low = [x for x in arr if x < pivot] 
    high = [x for x in arr if x > pivot] 
    pivots = [x for x in arr if x == pivot]  
    if k < len(low): 
        return median_of_medians(low, k) 
    elif k < len(low) + len(pivots): 
        return pivot 
    else: 
        return median_of_medians(high, k- len(low) - len(pivots))  
arr = [12, 3, 5, 7, 4, 19, 26] 
k = 3 
result = median_of_medians(arr, k-1) 
print("k-th smallest element:", result)

#2 median of medians
def median_of_medians(arr, k): 
    if len(arr) <= 5: 
        return sorted(arr)[k-1]      
    sublists = [arr[i:i + 5] for i in range(0, len(arr), 5)] 
    medians = [sorted(sublist)[len(sublist) // 2] for sublist in sublists]
    pivot = median_of_medians(medians, len(medians) // 2)  
    low = [x for x in arr if x < pivot] 
    high = [x for x in arr if x > pivot] 
    pivots = [x for x in arr if x == pivot]  
    if k < len(low): 
        return median_of_medians(low, k) 
    elif k < len(low) + len(pivots): 
        return pivot 
    else: 
        return median_of_medians(high, k- len(low) - len(pivots))  
arr = [23,17,31,44,55,21,20,18,19,27]
k = 5
result = median_of_medians(arr, k) 
print("k-th smallest element:", result)

#3 closest pair 
import heapq
def closestpair(points, k):
    distances = [(x**2 + y**2, [x, y]) for x, y in points]    
    closest_points = heapq.nsmallest(k, distances)    
    result = [point for distance, point in closest_points]    
    return result
points = [[3, 3], [5, -1], [-2, 4]]
k = 2
print(closestpair(points, k)) 

#4 closest pair tuples
def count(A,B,C,D):
    sum={}
    for a in A:
        for b in B:
            s=a+b
            if s in sum:
                sum[s]+=1
            else:
                sum[s]=1
    coun=0
    for c in C:
        for d in D:
            s=-(c+d)
            if s in sum:
                coun+=sum[s]
    return coun
A, B, C, D = [1, 2], [-2, -1],  [-1, 2],  [0, 2]
print(count(A,B,C,D))

