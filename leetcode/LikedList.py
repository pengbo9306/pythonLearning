# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        ha, hb = headA, headB
        while ha != hb:
            ha = ha.next if ha else headB
            hb = hb.next if ha else headA
        return ha

    def reverseList(self, head: ListNode) -> ListNode:
        pre = None
        cur = head
        while cur:
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        return pre


solution = Solution()
head = ListNode(4)
node1 = ListNode(3)
node2 = ListNode(2)
node3 = ListNode(1)
head.next = node1
node1.next = node2
node2.next = node3
result = solution.reverseList(head)

while result:
    print(result.val)
    result = result.next
