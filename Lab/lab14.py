#1 n queens
def queens(n):
    count = 0
    board = [-1] * n  
    def is_safe(row, col):
        for prev_row in range(row):
            prev_col = board[prev_row]
            if (prev_col == col or prev_col - col == prev_row - row or prev_col - col == row - prev_row): 
                return False
        return True

    def backtrack(row):
        nonlocal count
        if count == n:
            return True 
        if row == n:
            print(f"Solution {count + 1}:")
            for r in range(n):
                line = ["Q" if board[r] == c else "." for c in range(n)]
                print(" ".join(line))
            print()
            count += 1
            return False 
        for col in range(n):
            if is_safe(row, col):
                board[row] = col
                if backtrack(row + 1):
                    return True
                board[row] = -1

        return False

    backtrack(0)
n = 4
queens(n)

