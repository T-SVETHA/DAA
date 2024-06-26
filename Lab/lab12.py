#1 binomial coefficient
def fact(n):
    f=1
    for i in range (1,n+1):
        f=f*i
    return f
n=int(input())
k=int(input())
a=n-k
re=fact(n)//(fact(k)*fact(a))
print(re)

#2 Word wrap  
def sol(solution, n):
    if n == 0:
        return
    sol(solution, solution[n])
    print(f"Line : words :{solution[n]+1} to {n}")
def wordwrap(words, maxwidth):
    n = len(words)
    inf = float('inf')
    extras = [[0] * n for _ in range(n)]
    cost = [inf] * (n + 1)
    solution = [0] * (n + 1)
    for i in range(n):
        extras[i][i] = maxwidth - words[i]
        for j in range(i + 1, n):
            extras[i][j] = extras[i][j - 1] - words[j] - 1
    cost[0] = 0
    for j in range(1, n + 1):
        for i in range(j):
            if extras[i][j - 1] >= 0:
                current_cost = cost[i] + extras[i][j - 1] ** 2
                if current_cost < cost[j]:
                    cost[j] = current_cost
                    solution[j] = i
    sol(solution, n)

words = [3, 2, 4, 5]
maxwidth = 6
wordwrap(words, maxwidth)
