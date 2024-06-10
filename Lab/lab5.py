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



mat = [[1, 3, 11],[2, 4, 6]]
k = 5
ar = [0]
for i in mat:
    ar = sorted(a + b for a in ar for b in i)[:k]
output = ar[-1]
print(output)
