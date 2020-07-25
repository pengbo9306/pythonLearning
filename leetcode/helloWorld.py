# encoding: utf-8
from typing import List


class Solution:
    # 双指针
    def twoSum(self, numbers: List[int], target: int) -> List[int]:

        if not numbers:
            return []
        n = len(numbers)
        left = 0
        right = n - 1
        while left < right:
            if numbers[left] + numbers[right] == target:
                return [left + 1, right + 1]
            elif numbers[left] + numbers[right] < target:
                left = left + 1
            else:
                right = right - 1
        return [-1, -1]


solution = Solution()
print(List.__module__)
print(Solution.__module__)
# 双指针
result = solution.twoSum([2, 7, 11, 15], 9)
print(result)
print('hello world')
