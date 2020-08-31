# encoding: utf-8

# 广度优先搜索
from typing import List


class Solution:
    # 矩阵对角线最短距离
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        n = len(grid)
        if not grid or grid[0][0] == 1 or grid[1][1] == 0:
            return -1
        elif n <= 2:
            return n

        queue = [(0, 0, 1)]
        while queue:
            x, y, step = queue.pop(0)
            for dx, dy in [(-1, -1), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1)]:
                if x + dx == n - 1 and y + dy == n - 1:
                    return step + 1
                if 0 <= x + dx < n and 0 <= y + dy < n and grid[x + dx][y + dy] == 0:
                    queue.append((x + dx, y + dy, step + 1))
                    grid[x + dx][y + dy] = 1
        return -1;


solution = Solution()
print(solution.shortestPathBinaryMatrix([[0, 0, 0], [1, 1, 0], [1, 1, 0]]))
