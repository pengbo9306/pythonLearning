# encoding: utf-8
import heapq
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

    # 推排序
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return heapq.nlargest(k, nums)[-1]

    # 推排序

    def myFindKthLargest(self, nums: List[int], k: int) -> int:

        def adjust_heap(idx, max_length):
            l = 2 * idx + 1
            r = 2 * idx + 2
            max_idx = idx
            if r < max_length and nums[max_idx] < nums[r]:
                max_idx = r
            if l < max_length and nums[max_idx] < nums[l]:
                max_idx = l
            if max_idx != idx:
                nums[idx], nums[max_idx] = nums[max_idx], nums[idx]
                adjust_heap(max_idx, max_length)

        # 建堆
        length = len(nums)
        for i in range(length // 2 - 1, -1, -1):
            adjust_heap(i, length)
        print('初始堆：{}'.format(nums))
        # 排序(删除元素)
        for i in range(1, k + 1):
            nums[0], nums[- i] = nums[- i], nums[0]
            print('第{}次排序结果：{}'.format(i, nums))
            res = nums[-i]
            adjust_heap(0, length - i)
        return res


solution = Solution()
print(List.__module__)
print(Solution.__module__)
# 双指针
two_sum = solution.twoSum([2, 7, 11, 15], 9)
print('{}: {}'.format('two_sum', two_sum))
# 推排序
kth_largest = solution.findKthLargest([2, 1, 3, 9, 8], 5)
print('{}: {}'.format('kth_largest', kth_largest))
kth_largest = solution.myFindKthLargest([2, 1, 3, 9, 8], 5)

print('{}: {}'.format('kth_largest', kth_largest))
print('hello world')
