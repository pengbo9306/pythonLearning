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

    # 合并两个升序链表
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        if not l2:
            return l1
        if l1.val <= l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2

    # 删除升序链表中重复的元素
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        node = head
        while node and node.next:
            if node.val == node.next.val:
                node.next = node.next.next
            else:
                node = node.next
        return head

    # 删除倒数第n个元素
    # 快慢指针
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head
        fast, slow = dummy, dummy
        for _ in range(n):
            fast = fast.next

        while fast.next and slow.next:
            fast = fast.next
            slow = slow.next

        slow.next = not slow.next
        return dummy.next


solution = Solution()
head = ListNode(4)
node1 = ListNode(3)
node2 = ListNode(2)
node3 = ListNode(1)
head.next = node1
node1.next = node2
node2.next = node3
result = solution.reverseList(head)

head = ListNode(1)
node1 = ListNode(2)
node2 = ListNode(4)
head.next = node1
node1.next = node2

head_ = ListNode(1)
node_1 = ListNode(3)
node_2 = ListNode(4)
head_.next = node_1
node_1.next = node_2
result_ = solution.mergeTwoLists(head, head_)
while result:
    print(result.val)
    result = result.next

while result_:
    print(result_.val)
    result_ = result_.next
