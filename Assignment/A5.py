def add_matrix(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
def sub_matrix(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
def strassen(A, B):
    if len(A) == 1:
        return [[A[0][0] * B[0][0]]]
    mid = len(A) // 2
    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]
    
    B11 = [row[:mid] for row in B[:mid]]
    B12 = [row[mid:] for row in B[:mid]]
    B21 = [row[:mid] for row in B[mid:]]
    B22 = [row[mid:] for row in B[mid:]]

    P1 = strassen(add_matrix(A11, A22), add_matrix(B11, B22))
    P2 = strassen(add_matrix(A21, A22), B11)
    P3 = strassen(A11, sub_matrix(B12, B22))
    P4 = strassen(A22, sub_matrix(B21, B11))
    P5 = strassen(add_matrix(A11, A12), B22)
    P6 = strassen(sub_matrix(A21, A11), add_matrix(B11, B12))
    P7 = strassen(sub_matrix(A12, A22), add_matrix(B21, B22))
    
    C11 = add_matrix(sub_matrix(add_matrix(P1, P4), P5), P7)
    C12 = add_matrix(P3, P5)
    C21 = add_matrix(P2, P4)
    C22 = add_matrix(sub_matrix(add_matrix(P1, P3), P2), P6)
    C = []
    for i in range(mid):
        C.append(C11[i] + C12[i])
    for i in range(mid):
        C.append(C21[i] + C22[i])        
    return C
A = [[1, 2, 3,4],[5, 6, 7, 8],[1,2,3,4],[5,6,7,8]]
B = [[1,2,1,3],[1,4,1,5],[1,6,1,7],[1,8,1,9]]
C = strassen(A, B)
for row in C:
    print(row)

print("\n")
#1
def m(a, b):
    h = []
    i, j = 0, 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            h.append(a[i])
            i += 1
        else:
            h.append(b[j])
            j += 1
    while i < len(a):
        h.append(a[i])
        i += 1
    while j < len(b):
        h.append(b[j])
        j += 1
    return h
a = [1,2,4]
b = [1,3,4]
print(m(a,b))

print("\n")
#2
def m(l):
    def mt(a, b):
        h = []
        i, j = 0, 0
        while i < len(a) and j < len(b):
            if a[i] < b[j]:
                h.append(a[i])
                i += 1
            else:
                h.append(b[j])
                j += 1
        while i < len(a):
            h.append(a[i])
            i += 1
        while j < len(b):
            h.append(b[j])
            j += 1
        return h
    while len(l) > 1:
        ml = []
        for i in range(0, len(l), 2):
            l1 = l[i]
            l2 = l[i + 1] if (i + 1) < len(l) else []
            ml.append(mt(l1, l2))
        l = ml
    return l[0]
l = [[1,3,2,5], [5,26,9], [8,0,7]]
print(sorted(m(l)))

print("\n")
#3
def rem(a):
    if not a:
        return 0
    d = []
    for i in a:
        if i not in d:
            d.append(i)
    return d
a = [1,2,2,3,6,7,8,8,9]
print(rem(a))

print("\n")
#4
def s(a, t):
    l, h = 0, len(a) - 1
    while l <= h:
        m = (l + h) // 2
        if a[m] == t:
            return m
        if a[l] <= a[m]:
            if a[l] <= t < a[m]:
                h = m - 1
            else:
                l = m + 1
        else:
            if a[m] < t <= a[h]:
                l = m + 1
            else:
                h = m - 1
    return -1
a = [4,5,6,7,0,1,2]
t = 0
print(s(a,t))

print("\n")
#5
def fl(a,t):
    c=[]
    for i in range(len(a)):
        if a[i]==t:
            c.append(i)
        else:
            continue
    if not c:
        return [-1,-1]
    elif len(c)==1:
        return [c[0],-1]
    return c
a=[1,2,2,4,5,8,8]
t=9
print(fl(a,t))

print("\n")
#6
def col(ar):
    if len(ar)>1:
        mid = len(ar)//2
        l=col(ar[:mid])
        r=col(ar[mid:])
        i=j=k=0
        a=[0]*len(ar)
        while i<len(l) and j<len(r):
            if l[i]<r[j]:
                a[k]=l[i]
                i+=1
            else:
                a[k]=r[j]
                j+=1
            k+=1
        while i<len(l):
            a[k]=l[i]
            i+=1
            k+=1
        while j<len(r):
            a[k]=r[j]
            j+=1
            k+=1
        return a
    else:
        return ar
ar=[0,2,1,2,0,2,0,1]
print(col(ar))

print("\n")
#7
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def deleteDuplicates(self, head):
        if not head:
            return head       
        seen = set()
        seen.add(head.val)
        prev = head
        cur = head.next       
        while cur:
            if cur.val in seen:
                prev.next = cur.next
            else:
                seen.add(cur.val)
                prev = cur
            cur = cur.next       
        return head
head = ListNode(1)
head.next = ListNode(1)
head.next.next = ListNode(2)
head.next.next.next = ListNode(3)
head.next.next.next.next = ListNode(3)
solution = Solution()
new_head = solution.deleteDuplicates(head)
while new_head:
    print(new_head.val, end=" ")
    new_head = new_head.next


print("\n")
#8
def m(a1, m, a2, n):
    i, j, k = m-1, n-1, m+n-1
    while i >= 0 and j >= 0:
        if a1[i] > a2[j]:
            a1[k] = a1[i]
            i -= 1
        else:
            a1[k] = a2[j]
            j -= 1
        k -= 1
    while j >= 0:
        a1[k] = a2[j]
        j -= 1
        k -= 1
a1 = [1,2,3,0,0,0]
m1 = 3
a2 = [2,5,6]
n1 = 3
m(a1, m1, a2, n1)
print(a1)

print("\n")
#9
class T:
    def __init__(self, v=0, l=None, r=None):
        self.v = v
        self.l = l
        self.r = r
def c(a):
    if not a:
        return None
    m = len(a) // 2
    r = T(a[m])
    r.l = c(a[:m])
    r.r = c(a[m+1:])
    return r
def print_tree(node):
    if node:
        print(node.v, end=" ")
        print_tree(node.l)
        print_tree(node.r)
a1 = [-10, -3, 0, 5, 9]
t1 = c(a1) 
print_tree(t1)

print("\n")
#10
class L:
    def __init__(self, v=0, n=None):
        self.v = v
        self.n = n
def i(h):
    if not h or not h.n:
        return h
    d = L(0)
    c = h
    while c:
        t = c.n
        p = d
        while p.n and p.n.v < c.v:
            p = p.n
        c.n = p.n
        p.n = c
        c = t
    return d.n
def print_list(node):
    while node:
        print(node.v, end=" ")
        node = node.n
    print()
h1 = L(4, L(2, L(1, L(3))))
s1 = i(h1)  
print_list(s1)

print("\n")
#11
from collections import Counter
def s(f):
    c = Counter(f)
    return ''.join([k * v for k, v in c.most_common()])
s1 = "tree"
o1 = s(s1)  
print(o1)

print("\n")
#12
def m(a):
    mx, c = 0, 0
    for i, n in enumerate(a):
        mx = max(mx, n)
        if mx == i:
            c += 1
    return c
a1 = [4, 3, 2, 1, 0]
o1 = m(a1)  
print(o1)

print("\n")
#13
def i(a1, a2, a3):
    s1, s2, s3 = set(a1), set(a2), set(a3)
    return sorted(s1 & s2 & s3)
a1_1 = [1, 2, 3, 4, 5]
a2_1 = [1, 2, 5, 7, 9]
a3_1 = [1, 3, 4, 5, 8]
o1 = i(a1_1, a2_1, a3_1)  
print(o1)

print("\n")
#14
from collections import defaultdict
import heapq
def s(m):
    d = defaultdict(list)
    for i in range(len(m)):
        for j in range(len(m[0])):
            heapq.heappush(d[i-j], m[i][j])
    for i in range(len(m)):
        for j in range(len(m[0])):
            m[i][j] = heapq.heappop(d[i-j])
    return m
m1 = [[3, 3, 1, 1],[2, 2, 1, 2],[1, 1, 1, 2]]
o1 = s(m1)
print(o1)

