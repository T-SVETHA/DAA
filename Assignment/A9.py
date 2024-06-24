#1. Odd String Difference 
def odd(s): 
    return ''.join([c for i, c in enumerate(s) if i % 2 != 0]) 
 
s = "abcdefg" 
print(odd(s))   
 
#2. Words Within Two Edits of Dictionary 
def within(word, dic): 
    def iswithin(word1, word2): 
        if abs(len(word1) - len(word2)) > 2: 
            return False 
        i, j, edits = 0, 0, 0 
        while i < len(word1) and j < len(word2): 
            if word1[i] != word2[j]: 
                edits += 1 
                if edits > 2: 
                    return False 
                if len(word1) > len(word2): 
                    i += 1 
                elif len(word1) < len(word2): 
                    j += 1 
                else: 
                    i += 1 
                    j += 1 
            else: 
                i += 1 
                j += 1 
        edits += len(word1) - i + len(word2) - j 
        return edits <= 2 
 
    return [w for w in dic if iswithin(word, w)] 
dic= ["word", "ward", "world", "worm"] 
word = "word" 
print(within(word, dic))   
 
#3. Destroy Sequential 
def destroy(nums): 
    if not nums: 
        return 0 
    max_len = 1 
    current_len = 1 
    for i in range(1, len(nums)): 
        if nums[i] == nums[i - 1] + 1: 
            current_len += 1 
        else: 
            max_len = max(max_len, current_len) 
            current_len = 1 
    return max(max_len, current_len) 
 
nums = [1, 2, 3, 5, 6, 7, 9, 10, 11] 
print(destroy(nums))   
 
#4. Next Greater Element IV 
def next(nums): 
    res = [-1] * len(nums) 
    stack = [] 
    for i in range(len(nums)): 
        while stack and nums[stack[-1]] < nums[i]: 
            res[stack.pop()] = nums[i] 
        stack.append(i) 
    return res 
nums = [2, 1, 2, 4, 3] 
print(next(nums))   
 
#5. Average Value of Even Numbers That Are Divisible by Three 
def average(nums): 
    even= [num for num in nums if num % 2 == 0 and num % 3 == 0] 
    if not even: 
        return 0 
    return sum(even) / len(even) 
 
nums = [1, 2, 3, 6, 12, 15, 18] 
print(average(nums))   
 
#6. Most Popular video creator  
from collections import Counter 
def most(videos): 
    creatorviews = Counter() 
    for creator, views in videos: 
        creatorviews[creator] += views 
    maxviews = max(creatorviews.values()) 
    return [creator for creator, views in creatorviews.items() if views == maxviews] 
videos = [("creator1", 100), ("creator2", 200), ("creator1", 150), ("creator2", 50)] 
print(mostpopularcreator(videos)) 
 
#7. Minimum Addition to Make Integer 
def minimum(n): 
    steps = 0 
    while n % 2 != 0: 
        n += 1 
        steps += 1 
    return steps 
n = 15 
print(minimum(n))  
 
#8. Split Message Based on Limit 
def split(message, limit): 
    words = message.split() 
    result = [] 
    current = "" 
    for word in words: 
        if len(current) + len(word) + 1 <= limit: 
            current = current + " " + word if current else word 
        else: 
            result.append(current) 
            current= word 
    if current: 
        result.append(current) 
    return result 
message = "This is an example message that needs to be split based on a limit." 
limit = 10 
print(split(message, limit))  
