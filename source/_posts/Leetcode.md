---
title: Leetcode代码记录
tags: []
date: 2019-06-02 11:17:06
permalink:
categories: Algorithm
description: 面试算法题Python解答代码
image:
---
<p class="description"></p>

<img src="https://" alt="" style="width:100%" />

<!-- more -->
# 剑指offer

## 1. 二维数组中的查找

在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

**解题思路：**
对于target来说，从array的右上角进行搜索如果target比当前的数大则行加1列不变，向下搜索；如果target比当前的数小则列减1行不变，向左搜索。如果最后行列超出限制，则返回false。

```Python
# -*- coding:utf-8 -*-
class Solution:
    # array 二维列表
    def Find(self, target, array):
        row,col = 0,len(array[0])-1
        while row<=len(array)-1 and col>=0:
            if array[row][col]==target:
                return True
            elif target>array[row][col]:
                row += 1
            else:
                col -=1
        return False

```

## 2. 替换空格

请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

**可能有其他解法**

```Python
# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        print '%20'.join(s.split(' '))
```

## 3. 从尾到头打印链表

输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。

**解题思路**
使用递归，首先给出终止条件，如果链表为空返回[]，之后只需要return 当前链表的next+[当前的value]即可

```Python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        # write code here
        if listNode is None:
            return []
        return self.printListFromTailToHead(listNode.next)+[listNode.val]
```

## 4. 重建二叉树

根据二叉树的前序遍历和中序遍历的结果，重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

**解题思路：**
使用递归进行重构，首先给出终止条件，如果当前list是空返回none，如果长度只有1，返回当前值为根节点的tree，除了上面两个情况，首先构建root，是前序遍历的第一个值，之后根据index函数求解对应中序遍历list总改值节点的位置，之后使用递归，将前序遍历list和中序遍历list分别拆分作为左右节点

```Python
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if not pre or not tin:
            return None
        else:
            root = TreeNode(pre[0])
            idx = tin.index(pre[0])
            root.left = self.reConstructBinaryTree(pre[1:(idx+1)], tin[:idx])
            root.right = self.reConstructBinaryTree(pre[(idx+1):], tin[(idx+1):])
            return root
```


## 5. 用两个栈实现队列

用两个栈来实现一个队列，完成队列的 Push 和 Pop 操作。

**解题思路：**
构建两个两个list，push的话就不断压栈即可，pop的话如果list2有元素则弹出，否者将list1中的值全部弹出并压入list2中最后弹出一个

```Python
# -*- coding:utf-8 -*-
class Solution:
    '''使用两个栈，list的append和pop本身就是栈,记得初始化两个list
    如果有新值需要插入的话直接插入stack1中，如果有值要弹出的话，如果stack2中还有值，直接弹，否则
    把stack1中的值弹出到stack2中再弹出'''
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
    def push(self, node):
        self.stack1.append(node)
    def pop(self):
        if self.stack2:
            return self.stack2.pop()
        else:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
            return self.stack2.pop()
```


## 6. 旋转数组的最小数字

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

**解题思路：**
根据题目的思路，其实只要遍历数组找到对应的相邻的两个值是倒序的即可

```Python
# -*- coding:utf-8 -*-
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # write code here
        if len(rotateArray)==0:
            return 0
        for i in range(len(rotateArray)-1):
            if rotateArray[i]>rotateArray[i+1]:
                return rotateArray[i+1]
```


## 7. 斐波那契数列

大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。

**解题思路：**
fib数列主要使用递归表达式实现，只需要存储前后两个数字就可以进行更新，f(n) = f(n-1)+f(n-2)

```Python
# -*- coding:utf-8 -*-
class Solution:
    def Fibonacci(self, n):
        # write code here
        if n==0:
            return 0
        if n<=2:
            return 1
        i = 3
        before, after = 1, 1
        while i<=n:
            before,after = after, before+after
            i += 1
        return after
```


## 8. 跳台阶

一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。

**解题思路：**
依然使用递归思路，假设f(n)为当前台阶的跳法，因为只有两种跳跃方式，所以之前的一次跳跃方式只可能跳了1次或者跳了2次，如果跳了1次，那么其实和f(n-1)一样，如果跳了2次，那么其实和f(n-2)一样，所以可以通过递推公式f(n) = f(n-1)+f(n-2)进行更新

```Python
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloor(self, number):
        # write code here
        # f(n) = f(n-1) + f(n-2)
        if number==1:
            return 1
        if number==2:
            return 2
        i = 3
        before = 1
        after = 2
        while i<=number:
            before,after = after,before+after
            i += 1
        return after
```


## 9. 变态跳台阶

一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

**解题思路:**
其实跟上一题类似，只不过推广一下，相当于对之前所有的可能性求和，作为当前台阶的跳法，即f(n) = f(n-1)+f(n-2)...f(1),f(1)为1

```Python
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloorII(self, number):
        # write code here
        if number == 1:
            return 1
        if number ==2:
            return 2
        alist = [1,2]
        i = 3
        while i<=number:
            alist.append(sum(alist)+1)
            i += 1
        return alist[-1]
```


## 10. 矩形覆盖

我们可以用2 * 1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？

**解题思路：**
仍然使用上题的思路，因为当前的砖只能横着放，或者竖着放，因此上一次是竖着放的时候f(n)方法和f(n-1)方法相同，上一次是横着放的时候f(n)方法和f(n-2)方法相同，所以的得到递归式 f(n) = f(n-1) + f(n-2)

```Python
# -*- coding:utf-8 -*-
class Solution:
    def rectCover(self, number):
        # write code here
        # f(n) = f(n-1)+f(n-2)
        if number==0:
            return 0
        if number==1:
            return 1
        if number==2:
            return 2
        before,after = 1,2
        i = 3
        while i<=number:
            before,after = after, before+after
            i += 1
        return after
```

## 11. 二进制中1的个数

## 12. 数值的整数次方

给定一个 double 类型的浮点数 base 和 int 类型的整数 exponent，求 base 的 exponent 次方。

**解题思路：**
对于exponent为0/1/-1单独计算，之后判断正负设定flag，然后判断exponent%2是否为1，如果为1转化为base * [(base * base) ** (exponent/2)]，否则为(base * base) ** (exponent/2)

```Python
# -*- coding:utf-8 -*-
class Solution:
    def Power(self, base, exponent):
        # write code here
        if exponent==0:
            return 1
        elif exponent==1:
            return base
        elif exponent==-1:
            return 1/base
        else:
            isneg = False
            if exponent<0:
                isneg = True
            exponent = abs(exponent)
            if exponent%2==1:
                ret = self.Power(base*base,exponent/2)*base
            else:
                ret = self.Power(base*base,exponent/2)
            return 1/ret if isneg else ret
```

## 13. 调整数组顺序使奇数位于偶数前面

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

**解题思路**
使用list自带的insert以及pop，pop弹出对应的index的值，insert插入的位置可能比遍历数组的指针要慢（使用两个指针，一个指针遍历数组，一个指针用来标定奇数的位置）

```Python
# -*- coding:utf-8 -*-
class Solution:
    def reOrderArray(self, array):
        # write code here
        # 利用pop和insert
        idx = 0
        for k,v in enumerate(array):
            if v%2!=0:
                array.insert(idx,array.pop(k))
                idx += 1
        return array
```

## 14. 反转链表

输入一个链表，反转链表后，输出新链表的表头。

**解题思路：**
使用头插法，设置新链表，同时对于旧链表的next值改为新链表的next，并将新链表的next赋值为旧链表，注意需要更新旧链表，在最上层更新防止覆盖

```Python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        ret = ListNode(-1)
        while pHead:
            nextval = pHead.next
            pHead.next = ret.next
            ret.next = pHead
            pHead = nextval
        return ret.next
```

## 15. 合并两个排序的链表

输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

**解题思路：**
设置新链表，如果满足两个旧链表都不为空，则比较大小，并将小值赋给next，同时next更新，跳出循环之后，如果其中一个还不为空就直接加在新链表的next上即可

```Python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        ret = ListNode(0)
        tmp = ret
        while pHead1 and pHead2:
            if pHead1.val>=pHead2.val:
                tmp.next = pHead2
                pHead2 = pHead2.next
            else:
                tmp.next = pHead1
                pHead1 = pHead1.next
            tmp = tmp.next
        if pHead1:
            tmp.next = pHead1
        else:
            tmp.next = pHead2
        return ret.next
```

## 16. 数组中出现次数超过一半的数字

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。

**解题思路：**
多数投票问题，可以利用 Boyer-Moore Majority Vote Algorithm 来解决这个问题，使得时间复杂度为 O(N)。

使用 cnt 来统计一个元素出现的次数，当遍历到的元素和统计元素相等时，令 cnt++，否则令 cnt--。如果前面查找了 i 个元素，且 cnt == 0，说明前 i 个元素没有 majority，或者有 majority，但是出现的次数少于 i / 2 ，因为如果多于 i / 2 的话 cnt 就一定不会为 0 。此时剩下的 n - i 个元素中，majority 的数目依然多于 (n - i) / 2，因此继续查找就能找出 majority。

```Python
# -*- coding:utf-8 -*-
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        if not numbers:
            return 0
        val = numbers[0]
        cnt = 1
        for i in numbers[1:]:
            if i==val:
                cnt += 1
            else:
                cnt -= 1
            if cnt==0:
                val = i
                cnt = 1
        ret = 0
        for i in numbers:
            if i==val:
                ret += 1
        return val if ret>(len(numbers)/2) else 0
```

## 17. 最小的K个数

输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。

**解题思路：**
使用快排，把比当前数字小的数字作为left list，比当前数字大的数字作为right list，递归排序

```Python
# -*- coding:utf-8 -*-
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        def quickSort(x):
            if len(x)<=1:
                return x
            else:
                left = []
                right = []
                for i in x[1:]:
                    if i<=x[0]:
                        left.append(i)
                    else:
                        right.append(i)
                return quickSort(left) + [x[0]] + quickSort(right)
        if k>len(tinput):
            return []
        return quickSort(tinput)[:k]


def partition(arr):
    m = arr[0]
    l = [i for i in arr[1:] if i<=m]
    h = [i for i in arr[1:] if i>m]
    return l,m,h
def select(arr, k):
    l,m,h = partition(arr)
    if len(l)==k-1:
        return m
    elif len(l)<k-1:
        return select(h, k-len(l)-1)
    else:
        return select(l, k)
select([2,1,7,6,3,4,5], 5)

```


## 18. 连续子数组的最大和

HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)

**解题思路：**
如果list不存在返回0，长度为1返回第一个元素，否则对list进行遍历，分别设置两个变量，一个变量记录以当前元素结尾时最大和，一个记录整体最大值，使用tmpmax=max(tmpmax+i, i),maxval=max(maxval,tmpmax)，最后返回maxval即可

```Python
# -*- coding:utf-8 -*-
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        # write code here
        # f(n) = max(a[n], f(n-1)+a[n])
        # 动态规划
        alist = []
        if len(array)==1:
            return array[0]
        alist = [array[0],]
        for i in range(1,len(array)):
            tmp = max(array[i], alist[-1]+array[i])
            alist.append(tmp)
        return max(alist)
```

## 19. 整数中1出现的次数（从1到n整数中1出现的次数）

求出1 ~ 13的整数中1出现的次数,并算出100 ~ 1300的整数中1出现的次数？为此他特别数了一下1 ~ 13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。

**解题思路：**
对每个位置为1的个数进行加和，对每个位置上来说主要考虑三种情况，为0，为1，大于1，详细请看这个blog[思路](https://blog.csdn.net/weixin_40533355/article/details/83861895)
设置三个变量，cnt记录1个数，base记录当前指数位数，raw记录原始数字，
对于n不为0，n=n/10,single=n%10，之后对single进行判断，如果single为0则cnt+=n * (10 ** base)
如果为1，cnt+=n * (10 ** base)+raw%(10 ** base)+1
如果为其他，cnt+=(n+1) * (10 ** base)

```Python
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        # write code here
        cnt = 0
        base = 0
        raw = n
        while n!=0:
            single = n%10
            n = n/10
            if single==0:
                cnt += n*(10**base) # 0的时候只有上面的数字会影响
            elif single==1:
                cnt += n*(10**base) + raw%(10**base) + 1 # 1的时候上下都影响
            else:
                cnt += (n+1)*(10**base) # >1时上面影响+所有下面的
            base += 1
        return cnt
```


## 20. 把数组排成最小的数
输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

**解题思路：**
注意这里活用sorted的cmp比较器

```Python
# -*- coding:utf-8 -*-
class Solution:
    def PrintMinNumber(self, numbers):
        # write code here
        alist = map(str, numbers)
        ret = sorted(alist, cmp=lambda x,y: cmp(x+y,y+x)) # 如果x+y小输出x后y
        # ret = sorted(alist, cmp=lambda x,y: cmp(y+x,x+y)) # 如果x+y小输出y后x
        return ''.join(ret)
```


## 21. 第一个只出现一次的字符

在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.

**解题思路：**
利用字典，没啥说的，基本操作

```Python
# -*- coding:utf-8 -*-
class Solution:
    def FirstNotRepeatingChar(self, s):
        # write code here
        alist = []
        adic = {}
        for k,v in enumerate(s):
            if v in adic:
                adic[v] += 1
            else:
                adic[v] = 1
                alist.append((k,v)) # 活用元组保存位置
        for k,v in alist:
            if adic[v]==1:
                return k
        return -1
```


## 22. 数组中只出现一次的数字

一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。

**解题思路：**
使用字典，使用list，如果在就remove，最后剩下的就返回

```Python
# -*- coding:utf-8 -*-
class Solution:
    # 返回[a,b] 其中ab是出现一次的两个数字
    def FindNumsAppearOnce(self, array):
        # write code here
        adic = {}
        for i in array:
            if i in adic:
                adic[i] += 1
            else:
                adic[i] = 1
        alist = []
        for i in adic:
            if adic[i]==1:
                alist.append(i)
        return alist
```


## 23. 和为S的连续正数序列

小明很喜欢数学,有一天他在做数学作业时,要求计算出9 ~ 16的和,他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!

**解题思路：**
使用滑窗法，设定一个list存结果，start为1，end为2，计算start-end之间的和，如果start < end就while循环，如果和与target相同，list加入结果，如果大，start+1，如果小end+1同时更新和，直到最后返回list即可

注意：两个窗口都是从左边出发，不是两边夹逼。另外，当小于目标数时high++；大于目标数时low++，如果是high--，那么你仔细想想，你的窗口还怎么往后移动，整个结果在第一次大于目标数之后就不会往后移动，相反，而是在在这个low和high之间夹逼试探，最终啥都找不到或者只能找到一个。

```Python
# -*- coding:utf-8 -*-
class Solution:
    def FindContinuousSequence(self, tsum):
        # write code here
        alist = []
        idx_low,idx_high = 1, 2
        sum_val = (idx_low+idx_high)*(idx_high-idx_low+1)/2
        while idx_low<idx_high:
            if sum_val==tsum:
                alist.append(range(idx_low,idx_high+1))
                idx_high += 1 # 需要更新否则stall在这里了
                sum_val += idx_high # 这里只要加新的即可
            elif sum_val<tsum:
                idx_high += 1
                sum_val += idx_high # 这里只要加新的即可
            else:
                sum_val -= idx_low # 注意顺序
                idx_low += 1
        return alist
```

## 24. 翻转单词顺序列
牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？

```Python
# -*- coding:utf-8 -*-
class Solution:
    def ReverseSentence(self, s):
        # write code here
        return ' '.join(s.split(' ')[::-1])
```

## 25. 求1+2+3+...+n

求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

**解题思路：**
肯定是要用递归来解决，但是又不能使用if来给出终止条件，所以只能通过逻辑符号进行终止

```Python
# -*- coding:utf-8 -*-
class Solution:
    def Sum_Solution(self, n):
        # write code here
        return n>0 and self.Sum_Solution(n-1)+n
```

## 26. 数组中重复的数字

在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。

**解题思路：**
对数组进行遍历，对于每个位置，while判断当前index和value是否不相等，如果不等进入循环，判断value和list[value]相等则输出，不相等则替换位置，直至index和value相等。最后返回false

```Python
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        for k,v in enumerate(numbers):
            while k!=v:
                if v==numbers[v]:
                    duplication[0] = v
                    return True
                tmp = numbers[v]
                numbers[v] = v
                v = tmp
        return False
```


## 27. 构建乘积数组

给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0] * A[1] * ... * A[i-1] * A[i+1] * ... * A[n-1]。不能使用除法。

**解题思路：**
主要想法就是两次遍历，正序遍历每个元素都为前面所有元素之积，倒序遍历每个元素都为所有后面元素之积

```Python
# -*- coding:utf-8 -*-
class Solution:
    def multiply(self, A):
        # write code here
        ret = [1]
        tmp = 1
        for i in A[:-1]:
            tmp *= i
            ret.append(tmp)
        tmp = 1
        for i in range(len(A)-2,-1,-1):
            tmp *= A[i+1]
            ret[i] *= tmp
        return ret
```


## 28. 二叉树的下一个结点

给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。

**解题思路：**
因为是基于中序遍历，所以正常遍历是左中右，首先判断某个节点是否有右节点，如果有右节点设置临时节点tmp同时对tmp.left不断遍历，直到null然后返回对应值
对于没有有节点的值，向上找到父节点，判断是否父节点的left与当前节点是否相同，相同的话返回父节点，否则不断向上追溯，直到某个父节点的left与当前节点相同返回父节点

```Python
# -*- coding:utf-8 -*-
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None
class Solution:
    def GetNext(self, pNode):
        # write code here
        if pNode.right:
            tmp = pNode.right
            while tmp.left:
                tmp = tmp.left
            return tmp
        else:
            while pNode.next:
                parent = pNode.next
                if (parent.left==pNode):
                    return parent
                pNode = pNode.next
        return None
```


## 29. 矩阵中的路径

判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向上下左右移动一个格子。如果一条路径经过了矩阵中的某一个格子，则该路径不能再进入该格子。

**解题思路：**
    - 这道题利用回溯法，需要设定对应的矩阵的状态情况以及当前路径长度，需要一个辅助函数，主函数给定边界条件，当cols和rows都小于0或者要搜寻的路径小于0的时候返回False。
    - 因为不知道具体的其实点在哪，所以两层循环，对不同的ij进行试探hasPathCore(matrix, rows, cols, path, i, j, pathlen, markMTX)，其中pathlen是给定的路径的长度，markMTX是对应的状态。
    - hasPathCore这个函数首先判断pathlen是否跟len(path)相等，如果相等返回true。之后查看col/row是否满足边界条件，以及markMTX[i][j]状态是否未遍历，以及matrix[i][j]是否跟path[pathlen]相等，如果相等对当前矩阵状态进行变更同时pathlen+1，分别对上下左右进行递归or连接，如果返回true，直接return，如果返回false，对矩阵状态/pathlen进行退回，返回false

```Python
# -*- coding:utf-8 -*-
class Solution:
    def hasPath(self, matrix, rows, cols, path):
        # 过滤条件
        if len(path)==0 or not matrix or cols<0 or rows<0:
            return False
        markMTX = [0]*(rows*cols)
        pathlen = 0
        # 确定逐个起始点
        for i in range(rows):
            for j in range(cols):
                if self.hasPathCore(matrix, rows, cols, path, i, j, pathlen, markMTX):
                    return True
        return False
    def hasPathCore(self, matrix, rows, cols, path, row, col, pathlen, markMTX):
        # 3.终止条件
        if pathlen == len(path):
            return True
        # 2.设置初始防止一开始不满足
        haspath = False
        # 1.满足条件递归
        if row>=0 and row<rows and col>=0 and col<cols and matrix[cols*row+col]==path[pathlen] and not markMTX[cols*row+col]:
            pathlen += 1
            markMTX[cols*row+col] = 1
            haspath = self.hasPathCore(matrix, rows, cols, path, row-1, col, pathlen, markMTX) or \
                self.hasPathCore(matrix, rows, cols, path, row+1, col, pathlen, markMTX) or \
                self.hasPathCore(matrix, rows, cols, path, row, col-1, pathlen, markMTX) or \
                self.hasPathCore(matrix, rows, cols, path, row, col+1, pathlen, markMTX)
            if not haspath:
                pathlen -= 1
                markMTX[cols*row+col] = 0
        return haspath
```


## 30. 机器人的运动范围

地上有一个 m 行和 n 列的方格。一个机器人从坐标 (0, 0) 的格子开始移动，每一次只能向左右上下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于 k 的格子。

例如，当 k 为 18 时，机器人能够进入方格 (35,37)，因为 3+5+3+7=18。但是，它不能进入方格 (35,38)，因为 3+5+3+8=19。请问该机器人能够达到多少个格子？

**解题思路：**
边界判断，是否长宽都小于0，否则返回0
之后设置辅助函数，一个是求位数和的，一个是用来计算能有多少个符合的格子，设定矩阵状态，给定00为起始点，对于每次满足ij范围且state[i][j]为0且坐标位数之和小于阈值，则计算count=1+上下左右，返回count即可

```Python
# -*- coding:utf-8 -*-
class Solution:
    def movingCount(self, threshold, rows, cols):
        # write code here
        stateMTX = [0]*(rows*cols)
        num = self.GetNum(threshold, rows, cols, 0, 0, stateMTX)
        return num

    def GetNum(self, threshold, rows, cols, row, col, markmatrix):
        count = 0

        if self.GetSum(threshold, rows, cols, row, col, markmatrix):
            markmatrix[row * cols + col] = True
            count = 1 + self.GetNum(threshold, rows, cols, row - 1, col, markmatrix) + \
                    self.GetNum(threshold, rows, cols, row, col - 1, markmatrix) + \
                    self.GetNum(threshold, rows, cols, row + 1, col, markmatrix) + \
                    self.GetNum(threshold, rows, cols, row, col + 1, markmatrix)
        return count

    def GetSum(self, threshold, rows, cols, row, col, markmatrix):
        if row >= 0 and row < rows and col >= 0 and col < cols and self.getdigit(row) + self.getdigit(
                col) <= threshold and not markmatrix[row * cols + col]:
            return True
        return False
    def getdigit(self,number):
        sumval = 0
        while number>0:
            sumval += number%10
            number = number//10
        return sumval
```


### 31. 剪绳子(leetcode 343 整数拆分)

给定一个正整数 n，将其拆分为至少两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积。
你可以假设 n 不小于 2 且不大于 58。

n = 2
return 1 (2 = 1 + 1)

n = 10
return 36 (10 = 3 + 3 + 4)

**解题思路**
使用动态规划，当长度为2时返回1，当长度为3时返回2，当长度大于等于4时，因为当前最优解是建立在之前最优解的基础上的，因此，只需要遍历之前最优解进行乘积找到最大即可。首先设定list[0,1,2,3]因为按照分各状态前三个分各状态是这样的，之后只需要不断向list里面添加最大值即可。最后输出最后一个元素即可。

或者求与3的余数，如果为1减一个3错一个4，为2则单拎出2.

```Python
class Solution(object):
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n==0:
            return 0
        elif n<=2:
            return 1
        elif n==3:
            return 2
        else:
            alist = [0,1,2,3]
            idx = 4
            while idx<=n:
                maxval = 0
                for i in range(1,idx):
                    maxval = max(maxval, alist[i]*alist[idx-i])
                alist.append(maxval)
                idx += 1
            return alist[-1]
```


## 32. 在 O(1) 时间内删除链表节点
```
解题思路:
判断链表是否存在，然后判断删除节点的next是否是null，如果不是null，直接delete.val = next.val以及delete.next = next.next即可
如果是null要判断head是否与delete相等，如果相等直接head=None，否则产生一个tmp节点遍历直至tmp.next与删除节点相等，同时使tmp.next=null即可
```


### 33. 删除链表中重复的结点

**解题思路：**
对于基础情况如果phead为空或者next为空返回本身，否则设定nextnode，如果nextnode值与当前值不等，直接利用递归phead.next=self.deleteDuplication(phead.next)然后return-phead，否则使用while循环在满足有nextnode以及nextnode.val==phead.val条件时，不断.next，然后跳出循环时，返回利用递归返回self.deleteDuplication(nextnode)即可

```Python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteDuplication(self, pHead):
        # write code here
        if not pHead or not pHead.next:
            return pHead
        else:
            nextnode = pHead.next
            if nextnode.val==pHead.val:
                while nextnode and nextnode.val==pHead.val:
                    nextnode = nextnode.next
                return self.deleteDuplication(nextnode)
            else:
                pHead.next = self.deleteDuplication(pHead.next)
                return pHead
```


## 34. 链表中倒数第 K 个结点

**解题思路**
设链表的长度为 N。设置两个指针 P1 和 P2，先让 P1 移动 K 个节点，则还有 N - K 个节点可以移动。此时让 P1 和 P2 同时移动，可以知道当 P1 移动到链表结尾时，P2 移动到第 N - K 个节点处，该位置就是倒数第 K 个节点。

注意边界条件，k为负数head为空，还要考虑k>len(head)的情况，使用while循环避免

```Python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        if not head or k<0:
            return None
        p1,p2 = head,head
        i = 0
        while p1 and i<k:
            p1 = p1.next
            i += 1
        if i<k:
            return None
        while p1:
            p1 = p1.next
            p2 = p2.next
        return p2
```


## 35. 链表中环的入口结点

给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。

**解题思路：**
首先确定是否有环，快慢指针如果相等返回相遇节点。(一定在环里)
确定环的长度，根据该点在环内，如果相遇记录下长度
之后重新弄两个节点，想让一个走环长度，一个作为头，只后相同步数(1)相遇即为入口(因为此时环中点以及起始点距离环入口刚好都相等)

```Python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def EntryNodeOfLoop(self, pHead):
        # write code here
        '''
        判断是否为有环链表，如果是的话计算环的长度，指定两个idx，其中一个往后延伸环长度距离，之后以相同的步伐后移，
        当相交的时候对应的节点即为入口
        '''
        def isloop(pHead):
            # 判断是否为loop，是返回橡胶节点，一定在环中
            idx1,idx2 = pHead, pHead
            while idx1.next and idx2.next.next:
                idx1 = idx1.next
                idx2 = idx2.next.next
                if idx1 == idx2:
                    return idx1
            return False
        # 计算环长度
        ret = isloop(pHead)
        if ret:
            tmp = ret
            length = 1
            while tmp.next != ret:
                length += 1
                tmp = tmp.next
            # idx2先往后延长length距离，在以相同步伐前进，如果相等返回idx1/idx2节点
            idx1,idx2 = pHead, pHead
            for _ in range(length):
                idx2 = idx2.next
            while idx1!=idx2:
                idx1 = idx1.next
                idx2 = idx2.next
            return idx1
        return None
```


## 36. 树的子结构

**解题思路：**
设置一个辅助函数，叫做isRootSubtree，给定两个树，判断是不是这两个树相等，如果root1不存在或者root1.val!=root.val返回false，如果root2为None返回true，之后再利用isRootSubtree递归判断左右两个节点
对于主函数，如果root1或者root2为空返回false(因为空树不是子结构)，否则判断isRootSubtree以及对左右节点进行递归循环

```Python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        # write code here
        if not pRoot1 or not pRoot2:
            return False
        return self.is_subtree(pRoot1, pRoot2) or self.HasSubtree(pRoot1.left, pRoot2) or self.HasSubtree(pRoot1.right, pRoot2)
     
    def is_subtree(self, A, B):
        if not B:
            return True
        if not A or A.val != B.val:
            return False
        return self.is_subtree(A.left,B.left) and self.is_subtree(A.right, B.right)
```


## 37. 二叉树的镜像
操作给定的二叉树，将其变换为源二叉树的镜像。

**解题思路：**
首先边界条件，如果根节点是空的那么返回false，然后左右交换，再判断如果左节点存在则对于左节点递归，如果右节点存在则对于右节点递归

```Python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):
        # write code here
        '''还是使用递归，中间不变左右交换，因此先判断tree是不是空，而且左右是否全为空，如果是false停止，
        然后先换一次左和右，如果左子树非空再利用递归在左子树，右子树也是相同的'''
        if not root:
            return None
        root.left, root.right = root.right, root.left
        if root.left:
            self.Mirror(root.left)
        if root.right:
            self.Mirror(root.right)
```

## 38 对称的二叉树

请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

**解题思路：**
根节点如果为空返回true，否则使用辅助函数isequal来判断根节点左子节点与右节点是否相等
辅助函数是为了判断树是否是对称的，判断两个节点如果都空返回true，如果两个节点有一个空返回false，如果两个节点值相等再使用递归返回对左右子树分别判断and连接结果

```Python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def left_right(self, left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        if left.val==right.val:
            return self.left_right(left.left,right.right) and self.left_right(left.right,right.left)
        # class里面所有的function都需要使用self调用
    def isSymmetrical(self, pRoot):
        # write code here
        # 根节点比较特殊
        if not pRoot:
            return True
        return self.left_right(pRoot.left, pRoot.right)
```


## 39. 顺时针打印矩阵

矩阵顺时针打印结果为：1, 2, 3, 4, 8, 12, 16, 15, 14, 13, 9, 5, 6, 7, 11, 10

**解题思路：**
首先确定矩阵row和col的边界，使用while循环在row的上边界<=下边界和col左边界<=右边界时，
对矩阵顺时针遍历，一行一列之后需要判断row的上下边界是否相等，不相等遍历添加，同理col左右边界也是一样。最后统一对上下左右边界进行更新

```Python
# -*- coding:utf-8 -*-
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        # write code here
        ret = []
        c1,c2,r1,r2 = 0,len(matrix[0])-1,0,len(matrix)-1
        while c1<=c2 and r1<=r2:
            for i in range(c1,c2+1):
                ret.append(matrix[r1][i])
            for i in range(r1+1,r2+1):
                ret.append(matrix[i][c2])
            if r1!=r2:
                for i in range(c1,c2)[::-1]:
                    ret.append(matrix[r2][i])
            if c1!=c2:
                for i in range(r1+1,r2)[::-1]:
                    ret.append(matrix[i][c1])
            c1+=1;c2-=1;r1+=1;r2-=1
        return ret

```


## 40. 包含 min 函数的栈
定义栈的数据结构，请在该类型中实现一个能够得到栈最小元素的 min 函数。

**题目解析：**
对于整个class，设置初始值函数，设置push函数，如果help栈为空data和help压入，如果data压入的值小于help最后一个值，则将新值压入help，弹栈的时候help和data都弹，栈的最小值求解的时候直接弹出辅助栈最后一个值

```Python
# -*- coding:utf-8 -*-
class Solution:
    '''
    定义两个list分别是数据以及辅助列表，当有数值压入的时候，data一定压入，help判断是否为空，为空压入，
    不为空时如果最后一个值大于新值，也压入。弹栈的时候如果help和data相同则都弹出，否则只弹出data。
    top返回最后一个值即可。min返回help最后一个值即可。
    '''
    def __init__(self):
        self.data = []
        self.help = []
    def push(self, node):
        # write code here
        self.data.append(node)
        if not self.help:
            self.help.append(node)
        if self.help[-1]>node:
            self.help.append(node)
    def pop(self):
        # write code here
        if self.data[-1] == self.help[-1]:
            self.help.pop()
        return self.data.pop()
    def top(self):
        # write code here
        return self.data[-1]
    def min(self):
        # write code here
        return self.help[-1]
```


## 41. 栈的压入、弹出序列

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。
例如序列 1,2,3,4,5 是某栈的压入顺序，序列 4,5,3,2,1 是该压栈序列对应的一个弹出序列，但 4,3,5,1,2 就不可能是该压栈序列的弹出序列。

**解题思路：**
遍历压栈list，使用while循环如果压栈不空且压栈值与弹栈值第一个相等则弹出值，同时弹栈序列index向后推，最后判断压栈是否为空，为空返回true否则false

```Python
# -*- coding:utf-8 -*-
class Solution:
    def IsPopOrder(self, pushV, popV):
        # write code here
        '''
        解题思路：
        借助一个list和一个指针实现，list存储压栈元素，指针表征弹栈位置，
        遍历压栈，如果压栈之后最后一个元素和弹栈指针所指元素相等(list不空)，那么list弹栈，指针加1，
        如果不是则继续遍历压栈，最后查看list情况，如果空了返回True反之返回False
        '''
        alist = []
        j = 0
        for i in pushV: # 压栈只有一次，遍历即可
            alist.append(i)
            while alist and alist[-1]==popV[j]:
                alist.pop()
                j += 1
        if alist:
            return False
        return True
```


## 42 从上往下打印二叉树

从上往下打印出二叉树的每个节点，同层节点从左至右打印

**解题思路:**
使用广度优先搜索，考虑边界，如果根节点为空返回None，设置一个list放置节点，先放入根节点，如果list不空，则弹出节点同时加入节点对应的val，如果节点左右节点是有的之加入

```Python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        # write code here
        '''
        解题思路：
        通过每次只输出根节点的值，并把后续的值append到tmp中，直至tmp为空，遍历完成
        '''
        if not root:
            return []
        alist = []
        tmp = [root]
        while tmp:
            cur = tmp.pop(0)
            alist.append(cur.val)
            if cur.left:
                tmp.append(cur.left)
            if cur.right:
                tmp.append(cur.right)
        return alist
```


## 43. 把二叉树打印成多行

**解题思路：**
对于根节点进行判断，如果空则返回false，如果不空，加入临时list中，在list不为空时，使用一个辅助list保存当前层值，同时记录当前list长度，再利用一个循环对长度范围进行遍历，之后将list加入到结果列表

```Python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, pRoot):
        # write code here
        if not pRoot:
            return []
        else:
            ret = []
            tmp = [pRoot]
            while tmp:
                alist = []
                length = len(tmp)
                for _ in range(length):
                    node = tmp.pop(0)
                    alist.append(node.val)
                    if node.left:
                        tmp.append(node.left)
                    if node.right:
                        tmp.append(node.right)
                ret.append(alist)
            return ret
```


## 44. 按之字形顺序打印二叉树

请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。

**解题思路：**
和上题一样，列表控制元素添加，当前列表长度控制行数，同时再加一个cnt用来判断是否是基数行，如果是则正序否则倒序

```Python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Print(self, pRoot):
        # write code here
        if not pRoot:
            return []
        else:
            alist = [pRoot]
            ret = []
            cnt = 0
            while alist:
                length = len(alist)
                vallist = []
                cnt += 1
                for _ in range(length):
                    pop = alist.pop(0)
                    vallist.append(pop.val)
                    if pop.left:
                        alist.append(pop.left)
                    if pop.right:
                        alist.append(pop.right)
                if cnt%2==1:
                    ret.append(vallist)
                else:
                    ret.append(vallist[::-1])
            return ret
```


### 45. 二叉搜索树的后序遍历序列

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。假设输入的数组的任意两个数字都互不相同。

例如，下图是后序遍历序列 1,3,2 所对应的二叉搜索树。

**解题思路:**
对于二叉搜索树，左<中<右，后序遍历是左右中，因此，list最后一个是根节点，根据根节点的值将list分成两部分，左节点右节点，分割index，如果右节点的值中有小于根节点的，返回false，否则返回true。之后如果左节点不空递归查看，右节点一样，然后返回左右结果并值

```Python
# -*- coding:utf-8 -*-
class Solution:
    def VerifySquenceOfBST(self, sequence):
        # write code here
        '''
        解题思路：
        这里要输出False/True而不是yes/no
        注意二叉搜索树指的是二叉树中所有左边节点数值都比根节点数值小，右边节点数值都比根节点数值大
        后续遍历是先左节点再右节点最后根节点
        按照后序遍历方法root一定是最后一个值，左节点树在前右节点树在后，所以可以根据root值将list分为
        两部分，所以根据前部分值确定分隔点index(小于root)，再根据index确定时候后半部分值都大于root，
        如果是的话返回true，否则返回False，列表为空返回false
        '''
        if not sequence:
            return False
        root = sequence[-1]
        idx = 0
        for i in sequence[:-1]:
            if i<root:
                idx += 1
            else:
                break
        for j in sequence[idx:-1]:
            if j < root:
                return False
        return True
        if sequence[:idx+1]:
            left = self.VerifySquenceOfBST(sequence[:idx+1])
        if sequence[idx+1:-1]:
            right =  self.VerifySquenceOfBST(sequence[idx+1:-1])
        return left and right
```


## 46. 二叉树中和为某一值的路径

输入一颗二叉树和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。
下图的二叉树有两条和为 22 的路径：10, 5, 7 和 10, 12

**解题思路：**
使用递归，终止条件有两个如果tree为none返回none，如果node左右节点为空且当前节点值与target相等，返回[[node.val]]，然后对左右节点分别使用递归，最后遍历左右节点，分别于root.val相加返回最终值即可

```Python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        # write code here
        if not root:
            return [] # 终止条件
        if not root.left and not root.right and root.val==expectNumber:
            return [[root.val]] # 终止条件
        left = self.FindPath(root.left, expectNumber-root.val) # 循环体
        right = self.FindPath(root.right, expectNumber-root.val) # 循环体
        ret = []
        for i in left+right:
            ret.append([root.val]+i)
        return ret
```


## 47. 复杂链表的复制

输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的 head。

**解题思路：**
首先找到边界条件，如果为空返回None，之后复制链表分成三步，首先先复制node然后node与前后进行连接(next连接)利用while进行遍历，之后再对random进行复制(如果random存在的话)，最后再对链表进行分割(cur.next存在)，next=cur.next，之后cur.next=next.next，cur=next，之后再返回对应的clone节点即可

```Python
# -*- coding:utf-8 -*-
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None
class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        # write code here
        if not pHead:
            return None
        cur = pHead
        while cur:
            tmp = RandomListNode(cur.label)
            tmp.next = cur.next
            cur.next = tmp
            cur = tmp.next
        cur = pHead
        while cur:
            nextnode = cur.next
            if cur.random:
                nextnode.random = cur.random.next
            cur = nextnode.next
        cur = pHead
        clonehead = pHead.next
        while cur.next:
            nextval = cur.next
            cur.next = nextval.next
            cur = nextval
        return clonehead
```


## 48. 二叉搜索树与双向链表
输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

**解题思路：**
其实就是中序遍历的变体，首先初始化listhead和listtail分别为none，之后使用中序遍历，在中间过程中对节点进行判断，如果是第一次，就把root节点给head和tail，否则改变节点指向，tail.right指向root，root.left指向tail，同时tail更新为root，最后返回即可

```Python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def __init__(self):
        self.listhead = None
        self.listtail = None
    def Convert(self, pRootOfTree):
        # write code here
        if not pRootOfTree:
            return None
        self.Convert(pRootOfTree.left)
        if not self.listhead:
            self.listhead = pRootOfTree
            self.listtail = pRootOfTree
        else:
            self.listtail.right = pRootOfTree
            pRootOfTree.left = self.listtail
            self.listtail = pRootOfTree
        self.Convert(pRootOfTree.right)
        return self.listhead
```


## 49. 序列化二叉树

请实现两个函数，分别用来序列化和反序列化二叉树。

**解题思路：**
对于二叉树的序列化，可以对tree进行判断，如果为空返回#，否则返回根值,树左节点遍历,树右节点遍历，使用逗号隔开
对于反序列化，对序列按照逗号进行分割，然后使用一个辅助函数，辅助函数：如果list为空返回None，之后对对list.pop(0)，如果该值不为#则返回构建的树，记得左右节点是递归构建的，再return root即可

```Python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Serialize(self, root):
        # write code here
        # 前序遍历如果有none存在就返回'#'
        if not root:
            return '#'
        return str(root.val)+','+self.Serialize(root.left)+','+self.Serialize(root.right) # 字符串转化
    def deser(self, alist):
        if not alist: # 如果为空#都没了返回None
            return None
        val = alist.pop(0) # 如果有的话pop
        root = None # 设定为空，如果上来就是#，也要返回None
        if val!='#':
            root = TreeNode(int(val))
            root.left = self.deser(alist) # 顺序输出可以保证左右
            root.right = self.deser(alist)
        return root
    def Deserialize(self, s):
        # write code here
        alist = s.split(',')
        return self.deser(alist)
```


## 50. 字符串的排列
输入一个字符串，按字典序打印出该字符串中字符的所有排列。例如输入字符串 abc，则打印出由字符 a, b, c 所能排列出来的所有字符串 abc, acb, bac, bca, cab 和 cba。

**解题思路：**
首先基本的边界判断，如果ss为空返回[]，否则设定ret=[]作为结果存放，path=''作为开始的字符串长度
利用辅助函数，递归终止条件是当ss为空的对path插入到ret中(便利到最后一个字符的时候会一次性插入不用担心path有问题)，否则对每个字符进行遍历，使用递归更新子问题

ss是状态，ret是结果，类似于之前的机器人和路径查找，只不过是状态矩阵以及以及length分别作为判断，结果返回

```Python
# -*- coding:utf-8 -*-
class Solution:
    def Permutation(self, ss):
        # write code here
        return sorted(list(set(self.help(ss))))
    def help(self, ss):
        if len(ss)==0:
            return []
        if len(ss)==1:
            return [ss]
        else:
            ret = []
            for i,v in enumerate(ss):
                for j in self.help(ss[:i]+ss[i+1:]):
                    ret.append(v+j)
            return ret
```


## 51. 数据流中的中位数

如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。

**解题思路：**
可以初始化一个列表，然后insert数据流中的数，之后对于数据进行排序，在输出中位数即可
要熟悉一下各类排序

```Python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.alist = []
    def Insert(self, num):
        # write code here
        self.alist.append(num)
    def GetMedian(self,n=None):
        # write code here
        list_len = len(self.alist)
        sort_list = sorted(self.alist)
        if list_len%2!=0:
            return sort_list[list_len/2]
        else:
            return (sort_list[list_len/2]+sort_list[list_len/2-1])/2.0
```


## 52 字符流中第一个不重复的字符

请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符 "go" 时，第一个只出现一次的字符是 "g"。当从该字符流中读出前六个字符“google" 时，第一个只出现一次的字符是 "l"。

**解题思路：**
设置初始函数，一个s一个dic分别用作记录不重复字符以及对应的字符字典，插入功能：对字符进行统计，如果是新的需要s加入此元素(顺序加入)，之后对于不重复的元素只要遍历self.s进行对比是否为1即可

```Python
# -*- coding:utf-8 -*-
class Solution:
    # 返回对应char
    def __init__(self):
        self.s = ''
        self.adic = {}
    def FirstAppearingOnce(self):
        # write code here
        for i in self.s:
            if self.adic[i]==1:
                return i
        return '#'
    def Insert(self, char):
        # write code here
        if char in self.adic:
            self.adic[char] += 1
        else:
            self.adic[char] = 1
            self.s += char
```


## 53. 把数字翻译成字符串

给定一个数字，按照如下规则翻译成字符串：1 翻译成“a”，2 翻译成“b”... 26 翻译成“z”。一个数字有多种翻译可能，例如 12258 一共有 5 种，分别是 abbeh，lbeh，aveh，abyh，lyh。实现一个函数，用来计算一个数字有多少种不同的翻译方法。

**解题思路：**
典型的动态规划问题，分歧点就在于是否新加入的字符串能够和上一个字符串构成有效字母。
初始值，如果num<0返回0，如果len(str(num))为1，则返回1，否则对其进行倒叙遍历从倒数第二个开始，before,after=0,1,递推公式f(r-2) = f(r-1)+g(r-2,r-1)*f(r)；
before,after=after,after+flag*before

```Python
class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s[0]=='0':
            return 0
        else:
            ret = [1,1]
            for i in range(2,len(s)+1):
                if '10'<s[i-2:i]<='26' and s[i-2:i]!='20':
                    ret.append(ret[i-2]+ret[i-1])
                elif s[i-2:i]=='10' or s[i-2:i]=='20':
                    ret.append(ret[i-2])
                elif s[i-1]!='0':
                    ret.append(ret[i-1])
                else:
                    ret.append(0)
            return ret[len(s)]
https://zhuanlan.zhihu.com/p/60238439
```


## 54. 礼物的最大价值

在一个 m * n 的棋盘的每一个格都放有一个礼物，每个礼物都有一定价值（大于 0）。从左上角开始拿礼物，每次向右或向下移动一格，直到右下角结束。给定一个棋盘，求拿到礼物的最大价值。例如，对于如下棋盘

**解题思路：**
还是动态规划，maxval = max(left,up)+val[i][j]
首先进行边界判断，如果矩阵row和col都<=0，return 0
之后建立dp矩阵存储上一次的值，对row和col进行遍历，利用递推公式将dp进行填充，需要考虑left或者up是否存在(对index判断)，并对dp当前值进行更新，注意left和up要初始化为0
最后返回最后一个值即可


```Python
# -*- coding:utf-8 -*-
 
class Bonus:
    def getMost(self, board):
        # write code here
        if len(board)<=0 or len(board[0])<=0:
            return 0
        row,col=len(board),len(board[0])
        dp = [[0 for _ in range(col)] for _ in range(row)]
        for i in range(row):
            for j in range(col):
                up,left=0,0
                if i>0:
                    up = dp[i-1][j]
                if j>0:
                    left = dp[i][j-1]
                dp[i][j] = max(up,left)+board[i][j]
        return dp[row-1][col-1]
```


### 55. 丑数

把只包含因子 2、3 和 5 的数称作丑数（Ugly Number）。例如 6、8 都是丑数，但 14 不是，因为它包含因子 7。习惯上我们把 1 当做是第一个丑数。求按从小到大的顺序的第 N 个丑数。

**解题思路：**
对于index<7的直接返回index，之后初始化t2,t3,t5,对应list中的321，之后for循环直到n，向list不断加min(alist[t2]*2,alist[t3]*3,alist[t5]*5)即可
更新t2/t3/t5，分别都使用while循环，例如alist[t2]*2<=alist[-1], t2+=1
最后返回alist[-1]即可

```Python
# -*- coding:utf-8 -*-
class Solution:
    def GetUglyNumber_Solution(self, index):
        # write code here
        '''
        丑数是由235组成的，如果顺序排列好，新的丑数为之前所有丑数*2、3、5之后大于当前最大丑数的最小的值，
        但因为是排好序的，只需要确定T2，T3，T5的index这个index之前的数成2，3，5都小于当前最大丑数，只要更新这个index即可
        '''
        alist = range(1,7)
        if index<7:
            return index
        t2,t3,t5=3,2,1
        for i in range(6,index):
            alist.append(min([alist[t2]*2,alist[t3]*3,alist[t5]*5])) # 添加新丑数
            # 对t2t3t5更新
            while alist[t2]*2<=alist[-1]:
                t2 += 1
            while alist[t3]*3<=alist[-1]:
                t3 += 1
            while alist[t5]*5<=alist[-1]:
                t5 += 1
        return alist[-1]
```


## 56. 数字在排序数组中出现的次数

```Python
# -*- coding:utf-8 -*-
class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        num = 0
        if data:
            first = self.getFirstK(data, k , 0, len(data) - 1)
            last = self.getLastK(data, k, 0, len(data) - 1)
            if first > -1 and last > -1:
                num = last - first + 1
        return num

    def getFirstK(self, data, k, start, end):
        if start > end:
            return -1
        mid = (start + end) / 2
        midD = data[mid]
        if midD > k:
            end = mid - 1
        elif midD < k:
            start = mid + 1
        else:
            if (mid == 0) or (mid > 0 and data[mid - 1] != k):
                return mid
            else:
                end = mid - 1
        return self.getFirstK(data, k, start, end)

    def getLastK(self, data, k, start, end):
        if start > end:
            return -1
        mid = (start + end) / 2
        midD = data[mid]
        if midD > k:
            end = mid - 1
        elif midD < k:
            start = mid + 1
        else:
            if (mid == len(data) - 1) or (mid < len(data) - 1 and data[mid + 1] != k):
                return mid
            else:
                start = mid + 1
        return self.getLastK(data, k, start, end)#复制粘贴搞错了。。

```


## 57. 二叉查找树的第 K 个结点

利用二叉查找树中序遍历有序的特点。

**解题思路**
二叉树左边<中间<右边，所以使用中序遍历，使用递归，函数要有一个格外的list存结果
之后找到第k个元素即可

```Python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回对应节点TreeNode
    '''中序遍历可以对节点从大到小排'''
    def mid(self, root):
        if not root:
            return []
        ret = []
        if root.left:
            ret.extend(self.mid(root.left)) # extend是因为返回的是个list
        ret.append(root) # 这里是循环添加的元素
        if root.right:
            ret.extend(self.mid(root.right))
        return ret # 返回的是list
    def KthNode(self, pRoot, k):
        # write code here
        if k<=0: # k=0异常
            return None
        ret = self.mid(pRoot)
        if len(ret)<k: # k异常
            return None
        return ret[k-1]
#    def mid(self, pRoot):
#        if not pRoot:
#            return []
#        ret = []
#        ret += self.mid(pRoot.left)
#        ret.append(pRoot.val)
#        ret += self.mid(pRoot.right)
#        return ret
```


## 58. 二叉树的深度

输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。

**解题思路：**
递归：一棵树的长度 = 1 (根节点) + max(左子树长度, 右子树长度)
所以终止条件如果为空那么久返回0，否则使用递归左节点以及右节点且分别加1，然后判断两者大值

```Python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def TreeDepth(self, pRoot):
        # write code here
        '''
        递归
        '''
        if not pRoot:
            return 0
        return max(self.TreeDepth(pRoot.left)+1,self.TreeDepth(pRoot.right)+1)
```


## 59 平衡二叉树

问题描述：输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中的任意节点的左、右子树的深度相差不超过1，那么它就是一棵平衡二叉树。

**解题思路**
递归：使用上面的求深度函数，对每个节点的左右子树检查。
如果为空返回true，如果左右子树深度差大于1返回false，对左右节点进行递归检查

```Python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def IsBalanced_Solution(self, pRoot):
        # write code here
        if not pRoot:
            return True
        left = self.depth(pRoot.left)
        right = self.depth(pRoot.right)
        if abs(left-right)>1:
            return False
        return self.IsBalanced_Solution(pRoot.left) and self.IsBalanced_Solution(pRoot.right)
    def depth(self, pRoot):
        if not pRoot:
            return 0
        return max(1+self.depth(pRoot.left),1+self.depth(pRoot.right))
```


## 60. 和为 S 的两个数字

输入一个递增排序的数组和一个数字 S，在数组中查找两个数，使得他们的和正好是 S。如果有多对数字的和等于 S，输出两个数的乘积最小的。

**解题思路：**
使用双指针，对于左指针<右指针时while循环，如果对应的数字和为target，返回两个数字，此时肯定成绩最小，否则更新left和right的index

```Python
# -*- coding:utf-8 -*-
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        # write code here
        left,right = 0, len(array)-1
        while left<right:
            if array[left]+array[right]==tsum:
                return array[left],array[right]
            elif array[left]+array[right]>tsum:
                right -= 1
            else:
                left += 1
        return []
```


## 61. 滑动窗口的最大值

给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。

例如，如果输入数组 {2, 3, 4, 2, 6, 2, 5, 1} 及滑动窗口的大小 3，那么一共存在 6 个滑动窗口，他们的最大值分别为 {4, 4, 6, 6, 6, 5}。

**解题思路：**
直接遍历，求最大值即可

```Python
# -*- coding:utf-8 -*-
class Solution:
    def maxInWindows(self, num, size):
        # write code here
        if size==0:
            return []
        ret = []
        for i in range(len(num)-size+1):
            ret.append(max(num[i:i+size]))
        return ret
```


## 62. 扑克牌顺子

五张牌，其中大小鬼为癞子，牌面为 0。判断这五张牌是否能组成顺子。

**解题思路：**
首先统计0的个数，然后对数组排序，从不为0的数开始进行遍历，查看是否相邻数字相等，相等返回false，不相等则cnt-=后-前-1，如果最后cnt>=0，返回false

```Python
# -*- coding:utf-8 -*-
class Solution:
    def IsContinuous(self, numbers):
        # write code here
        if len(numbers)<5:
            return None
        sortnum = sorted(numbers)
        cnt = 0
        for i in sortnum:
            if i==0:
                cnt += 1
        for i in range(cnt,len(sortnum)-1):
            if sortnum[i]==sortnum[i+1]:
                return False
            cnt -= sortnum[i+1]-sortnum[i]-1
        return cnt>=0
```


## 63. 股票的最大利润

可以有一次买入和一次卖出，买入必须在前。求最大收益。 

**解题思路：**
设定当前index之前的最小值(第一个)，和最大收益(初始为0)，对数组进行遍历从第二个开始，确定之前最小值，以及以该节点为结尾的值和之前最大值的max，返回最后的max

```Python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        '''
        解题思路：动态规划，f(n) = max(f(n-1), A[n]-min(A[:n-1]))
        当前天收入=max(前一天受益，当前天价值-之前所有天最小值)
        '''
        if len(prices)<=1:
            maxpre = 0
        else:
            minpre = prices[0]
            maxpre = 0
            for i in range(1,len(prices)):
                maxpre = max(prices[i]-minpre, maxpre)
                minpre = min(minpre, prices[i])
        return maxpre
```


## 64. 圆圈中最后剩下的数

让小朋友们围成一个大圈。然后，随机指定一个数 m，让编号为 0 的小朋友开始报数。每次喊到 m-1 的那个小朋友要出列唱首歌，然后可以在礼品箱中任意的挑选礼物，并且不再回到圈中，从他的下一个小朋友开始，继续 0...m-1 报数 .... 这样下去 .... 直到剩下最后一个小朋友，可以不用表演。

**解题思路：**
约瑟夫环，如果n=0时返回-1，如果n=1返回0，之后return (self.xxx(n-1,m)+m)%n
环为n,指定数为m，结果等于度为n-1的约瑟夫环+m % n即可

```Python
return (self.xxx(n-1,m)+m)%n
```


## 65. n 个骰子的点数

```Python
把 n 个骰子仍在地上，求点数和为 s 的概率。

解题思路：
使用动态规划，dp[i][j]=dp[i-1][j-1]+dp[i-1][j-2]+dp[i-1][j-3]+dp[i-1][j-4]+dp[i-1][j-5]+dp[i-1][j-6]
首先初始化一个矩阵，行是n，列是n*6，值全部为0，然后对于第一行进行初始化，前6个值是1
之后使用循环对于1-n来说(1这里代表2)，遍历n~(n+1)*6，动态规划对结果进行填充，最后返回dp[n-1]即可
```
[说明](https://blog.csdn.net/leokingszx/article/details/80794407)


## 66. 树中两个节点的最低公共祖先

(1) 二叉查找树
解题思路：
因为所有节点都不重复，而且按照中序遍历一定是顺序的，所以，首先确定特殊情况，如果root为空返回None，如果root.val<p.val and root.val<q.val，则说明是在右子树，则使用递归(root.right,p,q)
如果root.val>p.val and root.val>q.val，则说明是在左子树，则使用递归(root.left,p,q)，否则return root(此时一定不相等)

```Python
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return None
        if (root.val > p.val and root.val > q.val):
            return self.lowestCommonAncestor(root.left, p, q)
        if (root.val < p.val and root.val < q.val):
            return self.lowestCommonAncestor(root.right, p, q)
        return root
```

(2) 普通二叉树

```Python
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if not root or p==root or q==root: # 判断特殊情况
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right =  self.lowestCommonAncestor(root.right, p, q)
        if left and right: # 如果左右节点都存在，返回root
            return root
        if left: # 否则，如果左节点都存在，返回left
            return left
        if right: # 否则，如果右节点都存在，返回right
            return right
```


## 67.在排序数组中查找数字
```Python
问题描述：统计一个数字在排序数组中出现的次数。如，输入排序数组{1,2,3,3,3,3,4,5}和数字3，由于3在这个数组中出现了4次，因此输出4。

解题思路：
直接用字典
或者使用二分查找，分别找到头和尾，查头的时候，在k=data[mid]的位置，如果mid是第一位或者如果不是第一位其之前的数不等于k，那么返回mid，否则end为mid-1
查尾的时候，在k=data[mid]的位置，如果mid是最后一位或者如果不是最后一位其之后的数不等于k，那么返回mid，否则start为mid+1(start>end返回-1)
之后再使用end-start+1，如果返回-1则直接返回0
```

## 68. 数字序列中的某一位数字
```Python
题目描述
数字以 0123456789101112131415... 的格式序列化到一个字符串中，求这个字符串的第 index 位。

解题思路：
对于数字进行循环，并且不断累加数字长度，当数字长度大于index的时候输出str(i)[length-n]即可
```


## 69.最长不含重复字符的子字符串
```Python
问题描述：请从字符串中找出一个最长的不包含重复字符串的子字符串，计算该最长子字符串的长度。假设字符串中只包含‘a’~‘z’的字符。例如，在字符串“arabcacfr”中，最长的不含重复字符的子字符串是“acfr”，长度是4。

解题思路：
动态规划，首先设定三个变量，curmax,totalmax,字母表字典记录上次出现位置(初始-1)
对字符串进行遍历，如果某个字符位置是-1或者字符串位置和上次出现位置之差>cur那么cur+=1，否则cur=两次位置距离差
然后跟totalmax进行比较，进行更新，更新字典。最后返回长度
```


## 70.两个链表的第一个公共节点
```Python
问题描述：输入两个链表，找出它们的第一个公共结点。

解题思路：
如果while循环两个链表不相等的话持续循环，如果1链表到结尾了则让1链表连接到2上否则next，2链表也一样，最后输出共同链表
```




# Leetcode习题


## 1. Two Sum
Given an array of integers, return indices of the two numbers such that they add up to a specific target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:
Given nums = [2, 7, 11, 15], target = 9,
Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].

给定一个整型数组 nums 和一个目标值 target ，请找出数组中和为 target 的两个整数，并返回这两个数的数组下标。
假设：
1）每个输入只对应一个答案；
2）不允许重复使用数组中的数值。
```Python
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        adic = {}
        for k,v in enumerate(nums):
            if v in adic:
                return [adic[v],k]
            else:
                adic[target-v] = k
        return None
```

## 27. Remove Element
Given an array nums and a value val, remove all instances of that value in-place and return the new length.
Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.
The order of elements can be changed. It doesn't matter what you leave beyond the new length.

Example:
Given nums = [0,1,2,2,3,0,4,2], val = 2,
Your function should return length = 5, with the first five elements of nums containing 0, 1, 3, 0, and 4.
Note that the order of those five elements can be arbitrary.
It doesn't matter what values are set beyond the returned length.

给定一个数组 nums 和一个值 val ，请在原地移除所有数值等于 val 的元素，并返回移除后数组的新长度。
要求：
1）原地操作的意思是不允许开辟额外的数组空间，空间复杂度必须为O(1)；
2）数组中元素的顺序是可以改变的，也不需要考虑超出新长度后面的元素。

**解题思路1：**
本题难点在于不允许使用额外的数组空间，因此考虑使用两个下标指针 i，j（初始都指向数组头部）：
当 nums[j] 等于 val（需移除的元素），忽略当前这个元素，j++；
当 nums[j] 不等于 val（需保留的元素），用 nums[j] 的值覆盖 nums[i] 的值，i++，j++。
```Python
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        start = 0
        end = len(nums)-1
        while start<=end:
            if nums[start]==val:
                nums[start] = nums[end]
                end -= 1
            else:
                start += 1
        nums = nums[:start]
        return start
```
**解题思路2：**
另一种思路仍是采用两个指针，一个从头向后扫，另一个从尾向前扫，遇到和 val 相等的值就和数组尾部的元素交换或覆盖。（假如需要移除的元素很少时，这种思路需要赋值的次数更少）
```Python
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        count = 0 # 保留与val不同的元素个数
        for i in nums:
            if i!=val:
                nums[count] = i
                count += 1
        nums = nums[:count] # 注意这里count其实多加了1
        return count

```

## 35. Search Insert Position
Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.
You may assume no duplicates in the array.

Example1:
Input: [1,3,5,6], 5
Output: 2

Example2:
Input: [1,3,5,6], 2
Output: 1

Example3:
Input: [1,3,5,6], 7
Output: 4

Example4:
Input: [1,3,5,6], 0
Output: 0
给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
你可以假设数组中无重复元素。

**解题思路：**
因为数组是有序的，所以直观的想法是考察nums中相邻元素与target的关系，如果target在相邻元素范围内，进一步判断是否与左边的数相等，进而判断插入或所在位置，当然除此之外还需要考虑target在nums最大最小值之外的情况。

```Python
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        # 先判断两头
        if nums[0]>=target:
            return 0
        if nums[-1]<target:
            return len(nums)
        # 在判断中间
        for i in range(0,len(nums)-1):
            if (target>=nums[i]) and (target<=nums[i+1]):
                if target==nums[i]:
                    return i
                else:
                    return i+1
```

## 53.Maximum Subarray

Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

Example:

Input: [-2,1,-3,4,-1,2,1,-5,4],
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.

Follow up:
If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.

给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**解题思路：**
利用动态规划思想，我们假设f(n)为末尾下标为n的最优子序列的和，那么f(n-1)即为末尾下标为n-1的最优子序列和，A(n)为nums数组中下标为n的元素，我们来考察三者的关系，即为：f(n)=max( f(n-1)+A(n), A(n) )，解释一下，因为我们之前已经给过定义f(n)是以n为末尾的子序列，那么他只可能有两种情况，要么是f(n-1)所在的子序列加上A(n)作为f(n)，要么就是只有A(n)这一个元素(因为A(n)时一定要存在的)，所以我们只需要判断一下，哪种序列和最大，就保留这个最大值，作为以n为末尾的子序列的最大值。这样我们遍历整个数组，把每个结果存起来，最后比较出最大值，即为整体子序列的最大值(其下标所在位置即为最优子序列末尾下表位置)。

```Python
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        alist = [] # 存储从0~len(nums)-1为末尾下标的最优子序列值
        for k,v in enumerate(nums):
            if k==0:
                before = v
            else:
                after = max(before+v,v)
                before = after
            alist.append(before)
        return max(alist)
```

## 66.Plus One

Given a non-empty array of digits representing a non-negative integer, plus one to the integer.
The digits are stored such that the most significant digit is at the head of the list, and each element in the array contain a single digit.
You may assume the integer does not contain any leading zero, except the number 0 itself.

Example 1:
Input: [1,2,3]
Output: [1,2,4]
Explanation: The array represents the integer 123.

Example 2:
Input: [4,3,2,1]
Output: [4,3,2,2]
Explanation: The array represents the integer 4321.

给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。
最高位数字存放在数组的首位， 数组中每个元素只存储一个数字。
你可以假设除了整数 0 之外，这个整数不会以零开头。

```Python
class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        # 借助类型转换
        before = ''
        for i in digits:
            before += str(i)
        after = str(int(before)+1)
        ret = [int(i) for i in after]
        return ret
```

## 88. Merge Sorted Array
Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.
Note:
The number of elements initialized in nums1 and nums2 are m and n respectively.
You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements from nums2.
Example:
Input:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3

Output: [1,2,2,3,5,6]

给定两个有序整数数组 nums1 和 nums2，将 nums2 合并到 nums1 中，使得 nums1 成为一个有序数组。

说明:
初始时， nums1 和 nums2 的元素数量分别为 m 和 n。
假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。

**解题思路：**
通过双指针进行操作，因为nums1数组长度为m+n且nums1和nums2都是有序的，所以，我们分别把m和n作为nums1和nums2的指针(即指向末尾)，对两个数组遍历，终止条件是其中一个数组下标为0(即只要有一个数组遍历完就结束)，将指针所指的元素进行比较，将较大的数放在num1的末尾，同时较大数所在数组的指针减1，以此类推。最后如果m下标所在数组没有遍历完，则说明剩下的数字都比nums2的最小数字小，nums1数组不应做任何操作，如果nums2所在数组没有遍历完，那么说明nums2剩下的数字都比num1最小值小，此时需要把nums2剩下数字填充到nums1最前面相应位置即可。

```Python
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        while m>0 and n>0:
            if nums1[m-1]>=nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -= 1
            else:
                nums1[m+n-1] = nums2[n-1]
                n -= 1
        if n>0:
            nums1[:n] = nums2[:n]
```
## 118.Pascal's Triangle
Given a non-negative integer numRows, generate the first numRows of Pascal's triangle.
Note: In Pascal's triangle, each number is the sum of the two numbers directly above it.

Example:
Input: 5
Output:
[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]

给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。
注意：在杨辉三角中，每个数是它左上方和右上方的数的和。

输入: 5
输出:
[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]

**解题思路：**
官方的图片其实很清楚的解释了原理，即从第三行开始，每一组两边的数都是1，中间的数都是由上一层相邻元素相加获得的，因此如果想要获取当前行数组，只需要上一层数组即可，有点像斐波那契数列的感觉，因为题目要求输出所有行因此需要把每行保存起来。

```Python
class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        alist = []
        for i in range(1,numRows+1):
            if i == 1:
                alist.append([1])
            elif i == 2:
                alist.append([1, 1])
            else:
                before = alist[-1]
                after = [1]+[before[j]+before[j+1] for j in range(len(before)-1)]+[1]
                alist.append(after)
            print alist
        return alist
```

## 121.Best Time to Buy and Sell Stock 
Say you have an array for which the ith element is the price of a given stock on day i.
If you were only permitted to complete at most one transaction (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit.
Note that you cannot sell a stock before you buy one.

Example1:
Input: [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
             Not 7-1 = 6, as selling price needs to be larger than buying price.

Example2:
Input: [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.

给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
如果你最多只允许完成一笔交易（即买入和卖出一支股票），设计一个算法来计算你所能获取的最大利润。
注意你不能在买入股票前卖出股票。

**解题思路：**
动态规划，f(n) = max(f(n-1), A[n]-min(A[:n-1]))
当前天收入=max(前一天受益，当前天价值-之前所有天最小值)
```Python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices) <= 1:
            return 0
        else:
            max_profit = 0 # max of f(n)
            min_num = prices[0] # min(A[:n-1])
            before = 0 # f(n-1)
            for i in range(len(prices)-1):
                cur = prices[i+1] # A[n]
                after = max(before, cur-min_num) # f(n)
                print after
                before = after
                min_num = cur if cur<min_num else min_num # update min value
                max_profit = before if before>max_profit else max_profit
            return max_profit

```

## 119.Pascal's Triangle II
Given a non-negative index k where k ≤ 33, return the kth index row of the Pascal's triangle.
Note that the row index starts from 0.

In Pascal's triangle, each number is the sum of the two numbers directly above it.

在杨辉三角中，每个数是它左上方和右上方的数的和。
给定一个非负索引 k，其中 k ≤ 33，返回杨辉三角的第 k 行。


**解题思路:**
当前层可以通过上一层得到，因此每次保留上一层数即可，得到当前层之后，将当前层替换成before即可
```Python
class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        if rowIndex==0:
            return [1]
        elif rowIndex==1:
            return [1,1]
        else:
            before = [1,1]
            for _ in range(rowIndex - 1):
                mid = [before[i]+before[i+1] for i in range(len(before)-1)]
                after = [1] + mid + [1]
                before = after
            return before
```

## 122. 买卖股票的最佳时机 II
给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

示例 1:
输入: [7,1,5,3,6,4]
输出: 7
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。

示例 2:
输入: [1,2,3,4,5]
输出: 4
解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。
     因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。

示例 3:
输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。


**解题思路：**
与股票买卖1相比，主要的差别在于可以多次买卖，换句话说也就是“股票买卖1”问题每个数字只能进行一次操作，但是该问题每个数字可以进行多次操作(买与卖)。因此，每天先买入(不会亏)，查看下一天，如果会亏就把当天的卖了，就相当于赚了0元，如果下一天会赚那么就直接卖掉，再买入下一天的股票，以此类推。
```Python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        max_profit = 0
        for i in range(len(prices)-1):
            if prices[i]<prices[i+1]:
                max_profit += prices[i+1] - prices[i]
        return max_profit
```

## 26. Remove Duplicates from Sorted Array
Given a sorted array nums, remove the duplicates in-place such that each element appear only once and return the new length.
Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.
Example 1:
Given nums = [1,1,2],
Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively.
It doesn't matter what you leave beyond the returned length.
Example 2:
Given nums = [0,0,1,1,1,2,2,3,3,4],

给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。
不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。

**解题思路：**
通过两个下标index和i完成，index下标从0开始，i从1开始，i是用来遍历数组的，当遇到重复值就跳过，index不增加，如果遇到的是非重复值，index下标加1并对原数组进行修改，将值改为i对应的值，直至遍历完整个数组，最后得到的元素组的前index+1个元素就是不重复元素。
```python
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # index控制不重复数字，i对列表变进行遍历
        index = 0
        for i in range(1,len(nums)):
            if nums[i]!=nums[index]:
                index += 1
                nums[index] = nums[i]
        nums = nums[:index+1]
        return index+1
```

## 167. Two Sum II
给定一个已按照升序排列 的有序数组，找到两个数使得它们相加之和等于目标数。
函数应该返回这两个下标值 index1 和 index2，其中 index1 必须小于 index2。

说明:
返回的下标值（index1 和 index2）不是从零开始的。
你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。

示例:
输入: numbers = [2, 7, 11, 15], target = 9
输出: [1,2]
解释: 2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。

**解题思路：**
跟第一题twosum的思路一样
就是遍历整个数组，之后将每一个target与数组元素之差作为字典的key存起来，同时将该元素所在位置下标作为value存起来，在遍历过程中，如果遇到元素与之前建立的字典的key相同，即取出之前的元素对应的下标，同时和当前元素对应下标，输出出来

```Python
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        adic = {}
        for k,v in enumerate(numbers):
            if v in adic:
                return [adic[v]+1,k+1]
            else:
                adic[target-v] = k
```


## 169. 求众数

给定一个大小为 n 的数组，找到其中的众数。众数是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。
你可以假设数组是非空的，并且给定的数组总是存在众数。
示例 1:
输入: [3,2,3]
输出: 3
示例 2:
输入: [2,2,1,1,1,2,2]
输出: 2

**解题思路：**
没啥说的，主要注意sorted的用法，应该还有更快的方法

```Python
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        adic = {}
        for i in nums:
            if i in adic:
                adic[i] += 1
            else:
                adic[i] = 1
        return sorted(adic.items(), key=lambda x:x[1], reverse=True)[0][0]
```

## 189.旋转数组
给定一个数组，将数组中的元素向右移动 k 个位置，其中 k 是非负数。

示例 1:
输入: [1,2,3,4,5,6,7] 和 k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右旋转 1 步: [7,1,2,3,4,5,6]
向右旋转 2 步: [6,7,1,2,3,4,5]
向右旋转 3 步: [5,6,7,1,2,3,4]

**解题思路：** 
利用list的pop功能(类似于栈)对数据弹出，再从头插入即可

```Python
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        for _ in range(k):
            nums.insert(0, nums.pop())
```

## 217.存在重复元素
给定一个整数数组，判断是否存在重复元素。

如果任何值在数组中出现至少两次，函数返回 true。如果数组中每个元素都不相同，则返回 false。

示例 1:
输入: [1,2,3,1]
输出: true

示例 2:
输入: [1,2,3,4]
输出: false

示例 3:
输入: [1,1,1,3,3,4,3,2,4,2]
输出: true

**解题思路：**
利用集合不重复特性，根据长度进行比较

```Python
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if len(set(nums))!=len(nums):
            return True
        return False
```

## 219.存在重复元素 II
给定一个整数数组和一个整数 k，判断数组中是否存在两个不同的索引 i 和 j，使得 nums [i] = nums [j]，并且存在 i 和 j 的差的绝对值不大于k的组合。

示例 1:
输入: nums = [1,2,3,1], k = 3
输出: true

示例 2:
输入: nums = [1,0,1,1], k = 1
输出: true

示例 3:
输入: nums = [1,2,3,1,2,3], k = 2
输出: false

**解题思路：**
首先建立一个字典，将所有的元素作为key，元素对应的下标作为value，如果value有重复则直接加入相应key所在的list即可，之后遍历字典，如果字典key对应的value有重复值，计算重复值相邻数字之间的差的绝对值，如果绝对值最小值小于k，则证明存在这样一组下标符合条件，否则返回False(即不存在这样的组合)

```Python
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        adic = {} # 储存数组值及对应index
        for i,j in enumerate(nums):
            if j in adic:
                adic[j].append(i)
            else:
                adic[j] = [i]
        for i in adic:
            adic_v = adic[i]
            if len(adic_v)>1:
                adic_v_diffmin = min([abs(adic_v[j]-adic_v[j+1]) for j in range(len(adic_v)-1)]) # 求相邻index的差的绝对值的最小值
                if adic_v_diffmin<=k:
                    return True
        return False
```

## 268. 缺失数字
给定一个包含 0, 1, 2, ..., n 中 n 个数的序列，找出 0 .. n 中没有出现在序列中的那个数。

示例 1:
输入: [3,0,1]
输出: 2

示例 2:
输入: [9,6,4,2,3,5,7,0,1]
输出: 8

说明:
你的算法应具有线性时间复杂度。你能否仅使用额外常数空间来实现?

**解题思路：**
求解0-n所有数的和，然后减去nums的和，差的那个数就是缺失的数。

```Python
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        list_len = len(nums)
        total = (1+list_len)*list_len/2
        rest = total-sum(nums)
        return rest
```



## 283. 移动零
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。

示例:
输入: [0,1,0,3,12]
输出: [1,3,12,0,0]

说明:
必须在原数组上操作，不能拷贝额外的数组。
尽量减少操作次数。

**解题思路：**
使用双指针，其中一个指针real_idx代表非0数字下标，另一个指针iter_idx代表遍历数组的下标，连个下标都为0，从头开始遍历数组，当遇到数组为0的时候real_idx不变，当不为0的时候，交换real_idx所在下标的值以及iter_idx所在下标的值，则相当于把0后置，相对的非0值顺序没有改变，最后返回nums原数组即为所求。

```Python
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        real_idx = 0
        for iter_idx in range(len(nums)):
            if nums[iter_idx]!=0:
                nums[real_idx],nums[iter_idx] = nums[iter_idx],nums[real_idx]
                real_idx += 1
        return nums
```

## 414. 第三大的数
给定一个非空数组，返回此数组中第三大的数。如果不存在，则返回数组中最大的数。要求算法时间复杂度必须是O(n)。

示例 1:
输入: [3, 2, 1]
输出: 1
解释: 第三大的数是 1.

示例 2:
输入: [1, 2]
输出: 2
解释: 第三大的数不存在, 所以返回最大的数 2 .

示例 3:
输入: [2, 2, 3, 1]
输出: 1
解释: 注意，要求返回第三大的数，是指第三大且唯一出现的数。
存在两个值为2的数，它们都排第二。

**解题思路：**
遍历数组，可以借鉴求最大值的思路，只不过扩展一下，求解前三大的数字而已，利用判断语句判断nums当前的值的范围，如果比最大值大就替代，在最大值第二大值之间就替代第二大值，在第二大值及第三大值之间就替代第三大值，否则跳过。有一点需要注意的是赋值的顺序，不能有交叉影响下一个复制表达式。

```Python
class Solution(object):
    def thirdMax(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        max1, max2, max3 = float('-inf'),float('-inf'),float('-inf')
        for i in nums:
            if i in [max1,max2,max3]:
                continue
            if i>max1:
                max3 = max2 # 注意顺序
                max2 = max1
                max1 = i # 注意修改的值放最后
            elif i>max2:
                max3 = max2
                max2 = i
            elif i>max3:
                max3 = i
            else:
                continue
        if max3==float('-inf'):
            return max1
        else:
            return max3
```

## 448. 找到所有数组中消失的数字
给定一个范围在  1 ≤ a[i] ≤ n ( n = 数组大小 ) 的 整型数组，数组中的元素一些出现了两次，另一些只出现一次。
找到所有在 [1, n] 范围之间没有出现在数组中的数字。
您能在不使用额外空间且时间复杂度为O(n)的情况下完成这个任务吗? 你可以假定返回的数组不算在额外空间内。

示例:
输入:
[4,3,2,7,8,2,3,1]
输出:
[5,6]

**解题思路：**
因为长度和给定数组范围一致，所以可以通过对数组中各元素所代表的下标进行标负，进行元素缺失的确认，遍历完之后剩下的正元素下标即为缺失元素，注意需要使用绝对值，因为有可能出现重复的数据

```Python
class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        for i in nums:
            nums[abs(i)-1] = -abs(nums[abs(i)-1])
        return [k+1 for k,v in enumerate(nums) if v>0 ]
```

## 485. 最大连续1的个数
给定一个二进制数组， 计算其中最大连续1的个数。

示例 1:
输入: [1,1,0,1,1,1]
输出: 3
解释: 开头的两位和最后的三位都是连续1，所以最大连续1的个数是 3.

注意：
输入的数组只包含 0 和1。
输入数组的长度是正整数，且不超过 10,000。

**解题思路：**
确定nums数组中各个0点的位置，之后进行相邻值求差，得到最大值即为最长1个数

```Python
class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        index0 = [-1]+[k for k,v in enumerate(nums) if v==0]+[len(nums)]
        length_list = [index0[i+1]-index0[i]-1 for i in range(len(index0)-1)]
        return max(length_list)
```


## 509. 斐波那契数
斐波那契数，通常用 F(n) 表示，形成的序列称为斐波那契数列。该数列由 0 和 1 开始，后面的每一项数字都是前面两项数字的和。也就是：

F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
给定 N，计算 F(N)。

示例 1：
输入：2
输出：1
解释：F(2) = F(1) + F(0) = 1 + 0 = 1.

示例 2：
输入：3
输出：2
解释：F(3) = F(2) + F(1) = 1 + 1 = 2.

示例 3：
输入：4
输出：3
解释：F(4) = F(3) + F(2) = 2 + 1 = 3.

提示：
0 ≤ N ≤ 30

**解题思路：**
基于表达式进行推导，每次只保留前两个数，基于这两个数生成新的数，之后对这两个数进行更新，以此类推。
f(n+2) = f(n) + f(n+1)

```Python
class Solution(object):
    def fib(self, N):
        """
        :type N: int
        :rtype: int
        """
        if N==0:
            return 0
        elif N<=2:
            return 1
        else:
            before = 1
            after = 1
            n = 3
            while n<=N:
                before, after = after, before+after
                n += 1
            return after
```


## 532. 数组中的K-diff数对
给定一个整数数组和一个整数 k, 你需要在数组里找到不同的 k-diff 数对。这里将 k-diff 数对定义为一个整数对 (i, j), 其中 i 和 j 都是数组中的数字，且两数之差的绝对值是 k.

示例 1:
输入: [3, 1, 4, 1, 5], k = 2
输出: 2
解释: 数组中有两个 2-diff 数对, (1, 3) 和 (3, 5)。
尽管数组中有两个1，但我们只应返回不同的数对的数量。

示例 2:
输入:[1, 2, 3, 4, 5], k = 1
输出: 4
解释: 数组中有四个 1-diff 数对, (1, 2), (2, 3), (3, 4) 和 (4, 5)。

示例 3:
输入: [1, 3, 1, 5, 4], k = 0
输出: 1
解释: 数组中只有一个 0-diff 数对，(1, 1)。

注意:
数对 (i, j) 和数对 (j, i) 被算作同一数对。
数组的长度不超过10,000。
所有输入的整数的范围在 [-1e7, 1e7]。

**解题思路:**
这个题目主要的难度在于对题目的解读，其实根据k分成不同情况即可

```Python
class Solution(object):
    def findPairs(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        adic = {}
        for i in nums:
            if i in adic:
                adic[i] += 1
            else:
                adic[i] = 1
        cnt = 0
        if k<0: # 意外情况返回0
            return cnt
        elif k==0: # 0-diff只需要统计出现次数大于1的数字即可
            for k,v in adic.items():
                if v>1:
                    cnt += 1
            return cnt
        else: # 其余的只需要查看i-k是否在adic中即可,注意不用绝对只是因为(i,j)(j,i)算1对
            for i in adic:
                if i-k in adic:
                    cnt += 1
            return cnt
```



## 561. 数组拆分 I
给定长度为 2n 的数组, 你的任务是将这些数分成 n 对, 例如 (a1, b1), (a2, b2), ..., (an, bn) ，使得从1 到 n 的 min(ai, bi) 总和最大。

示例 1:
输入: [1,4,3,2]
输出: 4
解释: n 等于 2, 最大总和为 4 = min(1, 2) + min(3, 4).

提示:
n 是正整数,范围在 [1, 10000].
数组中的元素范围在 [-10000, 10000].

**解题思路：**
要是想两对两对数的最小值的和最大，其实从最大的两对数进行推演，如果在nums上找到两个数组成的对，使得这个对的最小值最大，那么一定是整个数组中的最大数以及第二大数，之后再确定剩下数组中数对最小值最大，那么其实就是整个数组的第三大数以及第四大数组成的数对，以此类推，总结起来其实就是对数组排序，同时找到下标为偶数的数字的和。

```Python
class Solution(object):
    def arrayPairSum(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ret = sum([v for k,v in enumerate(sorted(nums)) if k%2==0])
        return ret
```


## 21. 合并两个有序链表

将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

示例：
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4

**解题思路：**
链表结构进行解题(没掌握好)

```Python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if l1 is None:
            return l2
        if l2 is None:
            return l1
        ret = ListNode(0)
        cur = ret
        while l1 and l2:
            if l1.val>=l2.val:
                cur.next = l2
                l2 = l2.next
            else:
                cur.next = l1
                l1 = l1.next
            cur = cur.next # 记得向后推
        if l1:
            cur.next=l1
        if l2:
            cur.next=l2
        return ret.next
```

## 83. 删除排序链表中的重复元素

给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。

示例 1:
输入: 1->1->2
输出: 1->2

示例 2:
输入: 1->1->2->3->3
输出: 1->2->3

**解题思路:**
链表结构

```Python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """        
        ret = head # 注意python变量名都是引用!!!!!!!!!
        if head is None:
            return head
        while head and head.next:
            while head.next and head.val==head.next.val:
                head.next = head.next.next
            head = head.next
        return ret
```

## 566. 重塑矩阵
在MATLAB中，有一个非常有用的函数 reshape，它可以将一个矩阵重塑为另一个大小不同的新矩阵，但保留其原始数据。
给出一个由二维数组表示的矩阵，以及两个正整数r和c，分别表示想要的重构的矩阵的行数和列数。
重构后的矩阵需要将原始矩阵的所有元素以相同的行遍历顺序填充。
如果具有给定参数的reshape操作是可行且合理的，则输出新的重塑矩阵；否则，输出原始矩阵。

示例 1:
输入: 
nums = 
[[1,2],
 [3,4]]
r = 1, c = 4
输出: 
[[1,2,3,4]]
解释:
行遍历nums的结果是 [1,2,3,4]。新的矩阵是 1 * 4 矩阵, 用之前的元素值一行一行填充新矩阵。

示例 2:
输入: 
nums = 
[[1,2],
 [3,4]]
r = 2, c = 4
输出: 
[[1,2],
 [3,4]]
解释:
没有办法将 2 * 2 矩阵转化为 2 * 4 矩阵。 所以输出原矩阵。

注意：
给定矩阵的宽和高范围在 [1, 100]。
给定的 r 和 c 都是正数。



```Python
class Solution(object):
    def matrixReshape(self, nums, r, c):
        """
        :type nums: List[List[int]]
        :type r: int
        :type c: int
        :rtype: List[List[int]]
        """
        if r*c!=len(nums)*len(nums[0]):
            return nums
        else:
            tmp = []
            new = []
            for i in nums:
                for j in i:
                    tmp.append(j)
                    if len(tmp)==c:
                        new.append(tmp)
                        tmp = []
            return new
```

## 581. 最短无序连续子数组
给定一个整数数组，你需要寻找一个连续的子数组，如果对这个子数组进行升序排序，那么整个数组都会变为升序排序。
你找到的子数组应是最短的，请输出它的长度。

示例 1:
输入: [2, 6, 4, 8, 10, 9, 15]
输出: 5
解释: 你只需要对 [6, 4, 8, 10, 9] 进行升序排序，那么整个表都会变为升序排序。

说明 :
输入的数组长度范围在 [1, 10,000]。
输入的数组可能包含重复元素 ，所以升序的意思是<=。

**解题思路**
就是排序前后数据进行比较

```Python
class Solution(object):
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        alist = []
        for k,v in enumerate(sorted(nums)):
            if nums[k]!=v:
                alist.append(k)
        if len(alist)==0:
            return 0
        return max(alist) - min(alist) + 1
```
## 141. 环形链表
给定一个链表，判断链表中是否有环。

为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。

示例 1：
输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。

示例 2：
输入：head = [1,2], pos = 0
输出：true
解释：链表中有一个环，其尾部连接到第一个节点。

**解题思路：**
利用快慢指针，如果有环快指针一定会在某个时刻与慢指针相遇，否则快指针为空

```Python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head is None or head.next is None or head.next.next is None:
            return False
        slow = head.next
        fast = head.next.next
        while slow!=fast and fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        if slow == fast:
            return True
        else:
            return False
```

## 160. 相交链表
编写一个程序，找到两个单链表相交的起始节点。

示例 1：
输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
输出：Reference of the node with value = 8
输入解释：相交节点的值为 8 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。

示例 2：
输入：intersectVal = 2, listA = [0,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
输出：Reference of the node with value = 2
输入解释：相交节点的值为 2 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [0,9,1,2,4]，链表 B 为 [3,2,4]。在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。

示例 3：
输入：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
输出：null
输入解释：从各自的表头开始算起，链表 A 为 [2,6,4]，链表 B 为 [1,5]。由于这两个链表不相交，所以 intersectVal 必须为 0，而 skipA 和 skipB 可以是任意值。
解释：这两个链表不相交，因此返回 null。

注意：
如果两个链表没有交点，返回 null.
在返回结果后，两个链表仍须保持原有的结构。
可假定整个链表结构中没有循环。
程序尽量满足 O(n) 时间复杂度，且仅用 O(1) 内存。

**解题思路：**
判断两个链表长度，将较长的链表移动到与短链表相差的长度位置，进行遍历，如果两个链表相同返回当前链表，否则返回None

```Python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        l1,l2 = 0,0
        l1node,l2node = headA,headB
        while l1node:
            l1 += 1
            l1node = l1node.next
        while l2node:
            l2 += 1
            l2node = l2node.next
        if l1>l2:
            for _ in range(l1-l2):
                headA = headA.next
        else: 
            for _ in range(l2-l1):
                headB = headB.next
        while headA!=headB:
            headA,headB = headA.next,headB.next 
        if headA:
            return headA
        else:
            return None
```

## 605. 种花问题
假设你有一个很长的花坛，一部分地块种植了花，另一部分却没有。可是，花卉不能种植在相邻的地块上，它们会争夺水源，两者都会死去。
给定一个花坛（表示为一个数组包含0和1，其中0表示没种植花，1表示种植了花），和一个数 n 。能否在不打破种植规则的情况下种入 n 朵花？能则返回True，不能则返回False。

示例 1:
输入: flowerbed = [1,0,0,0,1], n = 1
输出: True

示例 2:
输入: flowerbed = [1,0,0,0,1], n = 2
输出: False

注意:
数组内已种好的花不会违反种植规则。
输入的数组长度范围为 [1, 20000]。
n 是非负整数，且不会超过输入数组的大小。
在真实的面试中遇到过这道题？


**解题思路：**
头尾加一个0，然后找三个连续的0出现的次数大于n即可

```Python
class Solution(object):
    def canPlaceFlowers(self, flowerbed, n):
        """
        :type flowerbed: List[int]
        :type n: int
        :rtype: bool
        """
        new_list = [0]+flowerbed+[0]
        cnt = 0
        for i in range(1,len(new_list)-1):
            if new_list[i]==0 and new_list[i-1]==0 and new_list[i+1]==0:
                new_list[i] = 1
                cnt += 1
        return cnt>=n
```
## 203. 移除链表元素
删除链表中等于给定值 val 的所有节点。

示例:
输入: 1->2->6->3->4->5->6, val = 6
输出: 1->2->3->4->5


**解题思路：**
转化为数组，再重新构建链表

```Python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        alist = []
        while head:
            if head.val != val:
                alist.append(head.val)
            head = head.next
        ret = ListNode(0)
        tmp = ret
        for i in alist:
            tmp.next = ListNode(i)
            tmp = tmp.next
        return ret.next
```

## 206. 反转链表

反转一个单链表。

示例:
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL

进阶:
你可以迭代或递归地反转链表。你能否用两种方法解决这道题？


**解题思路：**
将链表转化成list，在进行反转

```Python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        alist = []
        while head:
            if head.val==0:
                alist.append(head.val)
            if head.val:
                alist.append(head.val)
            head = head.next
        ret = ListNode(0)
        tmp = ret
        for i in alist[::-1]:
            tmp.next = ListNode(i)
            tmp = tmp.next
        return ret.next
```

## 198. 打家劫舍
你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
给定一个代表每个房屋存放金额的非负整数数组，计算你在不触动警报装置的情况下，能够偷窃到的最高金额。

示例 1:
输入: [1,2,3,1]
输出: 4
解释: 偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。

示例 2:
输入: [2,7,9,3,1]
输出: 12
解释: 偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     偷窃到的最高金额 = 2 + 9 + 1 = 12 。

**解题思路：**
典型的动态规划问题，dp[i] = max(dp[i-1], dp[i-2]+a[i])
所以只需要保存前两天的记录就能根据当天的记录计算当前最大收益，初始的两天算0即可

```Python
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        before = 0
        after = 0
        for i in nums:
            before,after = after, max(before+i, after)
        return after

```

## 202. 快乐数
编写一个算法来判断一个数是不是“快乐数”。

一个“快乐数”定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数变为 1，也可能是无限循环但始终变不到 1。如果可以变为 1，那么这个数就是快乐数。

示例: 
输入: 19
输出: true
解释: 
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1

**解题思路：**
对不是快乐数的进行储存判断(有可能重复)，对是的返回true

```Python
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        alist = []
        while n!=1:
            if n in alist:
                return False
            else:
                alist.append(n)
            tmp = 0 
            for i in str(n):
                tmp += int(i)**2
            n = tmp
        return True

```

## 204. 计数质数
统计所有小于非负整数 n 的质数的数量。

示例:
输入: 10
输出: 4
解释: 小于 10 的质数一共有 4 个, 它们是 2, 3, 5, 7 。

**解题思路：**
对于每个位置赋1，如果是质数就就对其平方之后的所有间隔为本数的赋值为0

```Python
class Solution(object):
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        prime = [1]*n
        prime[:2] = [0,0]
        for i in range(2,int(n**0.5)+1):
            if prime[i]==1:
                prime[i*i:n:i]=[0]*len(prime[i*i:n:i])
        return sum(prime)
```

## 234. 回文链表
请判断一个链表是否为回文链表。

示例 1:
输入: 1->2
输出: false

示例 2:
输入: 1->2->2->1
输出: true

**解题思路：**
转成list

```Python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        alist = []
        while head:
            alist.append(head.val)
            head = head.next
        return alist==alist[::-1]
```

## 628. 三个数的最大乘积
给定一个整型数组，在数组中找出由三个数组成的最大乘积，并输出这个乘积。

示例 1:
输入: [1,2,3]
输出: 6

示例 2:
输入: [1,2,3,4]
输出: 24

**解题思路：**
最大的三个数乘积以及最小两个数乘积与最大数乘积比较选择大的

```Python
class Solution(object):
    def maximumProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        sort_list = sorted(nums)
        return max(sort_list[-1]*sort_list[-2]*sort_list[-3], sort_list[0]*sort_list[1]*sort_list[-1])
```

## 237. 删除链表中的节点
请编写一个函数，使其可以删除某个链表中给定的（非末尾）节点，你将只被给定要求被删除的节点。

示例 1:
输入: head = [4,5,1,9], node = 5
输出: [4,1,9]
解释: 给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.

示例 2:
输入: head = [4,5,1,9], node = 1
输出: [4,5,9]
解释: 给定你链表中值为 1 的第三个节点，那么在调用了你的函数之后，该链表应变为 4 -> 5 -> 9.
 
说明:
链表至少包含两个节点。
链表中所有节点的值都是唯一的。
给定的节点为非末尾节点并且一定是链表中的一个有效节点。
不要从你的函数中返回任何结果。

**解题思路：**
注意只有当前要删节点的访问权限，所以直接删除就好。。

```Python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next
```

## 643. 子数组最大平均数 I
给定 n 个整数，找出平均数最大且长度为 k 的连续子数组，并输出该最大平均数。

示例 1:
输入: [1,12,-5,-6,50,3], k = 4
输出: 12.75
解释: 最大平均数 (12-5-6+50)/4 = 51/4 = 12.75

注意:
1 <= k <= n <= 30,000。
所给数据范围 [-10,000，10,000]。

**解题思路：**
计算的时候需要主要不能重复计算否则超时，可以使用减去头加上尾的方法

```Python
class Solution(object):
    def findMaxAverage(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: float
        """
        ret = float('-inf')
        for i in range(len(nums)-k+1):
            if i==0:
                mean_val = sum(nums[i:i+k])
            else:
                mean_val = mean_val-nums[i-1]+nums[i+k-1]
            if mean_val>ret:
                ret = mean_val
        return ret*1.0/k
```

## 876. 链表的中间结点
给定一个带有头结点 head 的非空单链表，返回链表的中间结点。

如果有两个中间结点，则返回第二个中间结点。

示例 1：
输入：[1,2,3,4,5]
输出：此列表中的结点 3 (序列化形式：[3,4,5])
返回的结点值为 3 。 (测评系统对该结点序列化表述是 [3,4,5])。
注意，我们返回了一个 ListNode 类型的对象 ans，这样：
ans.val = 3, ans.next.val = 4, ans.next.next.val = 5, 以及 ans.next.next.next = NULL.

示例 2：
输入：[1,2,3,4,5,6]
输出：此列表中的结点 4 (序列化形式：[4,5,6])
由于该列表有两个中间结点，值分别为 3 和 4，我们返回第二个结点。

提示：
给定链表的结点数介于 1 和 100 之间。

**解题思路：**
借助列表即可完成

```Python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        alist = []
        while head:
            alist.append(head.val)
            head = head.next
        ret = ListNode(0)
        tmp = ret
        for i in alist[len(alist)/2:]:
            tmp.next = ListNode(i)
            tmp = tmp.next
        return ret.next
```

## 876


## 13. 罗马数字转整数
罗马数字包含以下七种字符: I， V， X， L，C，D 和 M。

字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：

I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 
C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。
给定一个罗马数字，将其转换成整数。输入确保在 1 到 3999 的范围内。

示例 1:
输入: "III"
输出: 3

示例 2:
输入: "IV"
输出: 4

示例 3:
输入: "IX"
输出: 9

示例 4:
输入: "LVIII"
输出: 58
解释: L = 50, V= 5, III = 3.

示例 5:
输入: "MCMXCIV"
输出: 1994
解释: M = 1000, CM = 900, XC = 90, IV = 4.

**解题思路：**
从后向前遍历，如果小于前一个就加，如果大于前一个就减


```Python
class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        adic = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000,}
        revs = s[::-1]
        numsum = adic[revs[0]]
        for i,v in enumerate(revs[1:]):
            if adic[v]>=adic[revs[i]]:
                numsum += adic[v]
            else:
                numsum -= adic[v]
        return numsum
```


## 14. 最长公共前缀

编写一个函数来查找字符串数组中的最长公共前缀。
如果不存在公共前缀，返回空字符串 ""。

示例 1:
输入: ["flower","flow","flight"]
输出: "fl"

示例 2:
输入: ["dog","racecar","car"]
输出: ""
解释: 输入不存在公共前缀。
说明:

所有输入只包含小写字母 a-z 。

**解题思路：**
注意这里"前缀"这个描述，之后利用python的转置即可

```Python
class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        ret = ''
        for i in zip(*strs):
            if len(set(i))==1:
                ret += i[0]
            else:
                return ret
        return ret
```




## 661. 图片平滑器
包含整数的二维矩阵 M 表示一个图片的灰度。你需要设计一个平滑器来让每一个单元的灰度成为平均灰度 (向下舍入) ，平均灰度的计算是周围的8个单元和它本身的值求平均，如果周围的单元格不足八个，则尽可能多的利用它们。

示例 1:
输入:
[[1,1,1],
 [1,0,1],
 [1,1,1]]
输出:
[[0, 0, 0],
 [0, 0, 0],
 [0, 0, 0]]

解释:
对于点 (0,0), (0,2), (2,0), (2,2): 平均(3/4) = 平均(0.75) = 0
对于点 (0,1), (1,0), (1,2), (2,1): 平均(5/6) = 平均(0.83333333) = 0
对于点 (1,1): 平均(8/9) = 平均(0.88888889) = 0

注意:
给定矩阵中的整数范围为 [0, 255]。
矩阵的长和宽的范围均为 [1, 150]。

**解题思路：**
就是很简单的遍历，没啥特点，先确定左上角位置坐标以及右下角坐标，然后注意和数组的边界的关系，求均值即可

```Python
class Solution(object):
    def imageSmoother(self, M):
        """
        :type M: List[List[int]]
        :rtype: List[List[int]]
        """
        row = len(M)
        col = len(M[0])
        ret = []
        for i in range(row):
            tmp = []
            for j in range(col):
                left = [max(i-1,0),max(j-1,0)] # 左上角坐标
                right = [min(i+1,row-1),min(j+1,col-1)] # 右下角坐标
                sum_val = 0
                # 遍历需要平滑位置的数值
                for p in range(left[0],right[0]+1):
                    for q in range(left[1],right[1]+1):
                        # print p,q
                        sum_val += M[p][q]
                # 根据窗口的形状求均值
                avg_val = sum_val/((right[0]+1-left[0])*(right[1]+1-left[1]))
                tmp.append(avg_val)
            ret.append(tmp)
        return ret
        
```

## 20. 有效的括号
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
有效字符串需满足：
左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
注意空字符串可被认为是有效字符串。

示例 1:
输入: "()"
输出: true

示例 2:
输入: "()[]{}"
输出: true

示例 3:
输入: "(]"
输出: false

示例 4:
输入: "([)]"
输出: false

示例 5:
输入: "{[]}"
输出: true


**解题思路:**
利用栈进行操作，如果是左边的括号就压进去，右边的就弹出，判断最后是否为空即可

```Python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        adic = {'(':')', '[':']', '{':'}'}
        alist = []
        for i in s:
            if i in adic:
                alist.append(i)
            else:
                if (not alist) or adic[alist.pop()]!=i:
                    return False
        if alist:
            return False
        return True
```

##28. 实现strStr()

实现 strStr() 函数。
给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。

示例 1:
输入: haystack = "hello", needle = "ll"
输出: 2

示例 2:
输入: haystack = "aaaaa", needle = "bba"
输出: -1
说明:

当 needle 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。
对于本题而言，当 needle 是空字符串时我们应当返回 0 。这与C语言的 strstr() 以及 Java的 indexOf() 定义相符。

**解题思路：**
直接判断就可以了，没什么技巧

```Python
class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        lenN = len(needle)
        lenH = len(haystack)
        for i in range(lenH-lenN+1):
            if haystack[i:i+lenN]==needle:
                return i
        return -1
```


## 665. 非递减数列

给定一个长度为 n 的整数数组，你的任务是判断在最多改变 1 个元素的情况下，该数组能否变成一个非递减数列。
我们是这样定义一个非递减数列的： 对于数组中所有的 i (1 <= i < n)，满足 array[i] <= array[i + 1]。

示例 1:
输入: [4,2,3]
输出: True
解释: 你可以通过把第一个4变成1来使得它成为一个非递减数列。

示例 2:
输入: [4,2,1]
输出: False
解释: 你不能在只改变一个元素的情况下将其变为非递减数列。
说明:  n 的范围为 [1, 10,000]。

**解题思路：**
其实就是遇到nums[i]>nums[i+1]时，修改nums[i]还是nums[i+1]，如果在头上修改nums[i]，再根据i-1，i+1判断修改哪个值，i-1>i+1时修改i+1(不能保证i+1不变所以只能修改这个元素)，否则修改i-1保持i+1不变

```Python
class Solution(object):
    def checkPossibility(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        cnt = 0
        for i in range(len(nums)-1):
            if nums[i]>nums[i+1]:
                cnt += 1
                if i==0:
                    nums[i] = nums[i+1]
                else:
                    if nums[i-1]>nums[i+1]:
                        nums[i+1] = nums[i]
                    else:
                        nums[i] = nums[i+1]
            if cnt>1:
                return False
        return True
```

## 38. 报数

报数序列是一个整数序列，按照其中的整数的顺序进行报数，得到下一个数。其前五项如下：
1.     1
2.     11
3.     21
4.     1211
5.     111221
1 被读作  "one 1"  ("一个一") , 即 11。
11 被读作 "two 1s" ("两个一"）, 即 21。
21 被读作 "one 2",  "one 1" （"一个二" ,  "一个一") , 即 1211。

给定一个正整数 n（1 ≤ n ≤ 30），输出报数序列的第 n 项。

注意：整数顺序将表示为一个字符串。

示例 1:
输入: 1
输出: "1"

示例 2:
输入: 4
输出: "1211"

**解题思路：**
只要保留上一次的值就可以长生出当前值，需要有几个变量记录，i当前正整数，val，cnt使用来统计数字的值以及个数的(转化成字符串统计)，strNum当前的报数值。说白了这道题就是一个统计数字的题。

```python
class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        if n==1:
            return '1'
        pre = '1' # 用来记录之前时刻的数字，用来推断下一个数字
        i = 1 # 当前数字值
        while i<n:
            val = pre[0] # 当前的值
            cnt = 0 # 当前值的个数
            strNum = '' # 产生出来的数字
            for j in pre:
                if j==val:
                    cnt += 1
                else:
                    strNum += str(cnt)+val # 拼接
                    cnt = 1
                    val = j
            strNum += str(cnt)+val
            pre = strNum
            i += 1
        return pre
```

## 58. 最后一个单词的长度

给定一个仅包含大小写字母和空格 ' ' 的字符串，返回其最后一个单词的长度。
如果不存在最后一个单词，请返回 0 。

说明：一个单词是指由字母组成，但不包含任何空格的字符串。

示例:
输入: "Hello World"
输出: 5


```Python
class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        return len(s.strip().split(' ')[-1])
```


## 674. 最长连续递增序列
给定一个未经排序的整数数组，找到最长且连续的的递增序列。

示例 1:
输入: [1,3,5,4,7]
输出: 3
解释: 最长连续递增序列是 [1,3,5], 长度为3。
尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，因为5和7在原数组里被4隔开。 

示例 2:
输入: [2,2,2,2,2]
输出: 1
解释: 最长连续递增序列是 [2], 长度为1。
注意：数组长度不会超过10000。

**解题思路：**
首先肯定是要设置滑动窗口的，比较nums[i]和nums[i+1]的值的大小，同时设置两个变量本别记录当前最大子序列长度，和全局最大子序列长度，每次先判断是否nums[i]<nums[i+1]，如果为真，当前的长度+1也就是tmp+1，同时跟全剧最长子序列比较如果大于全局就更新，如果nums[i]>=nums[i+1]，则重置当前子序列长度即可。注意序列为空的情况

```Python
class Solution(object):
    def findLengthOfLCIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        tmp = 1 # 存储当前的升序值
        ret = 1 # 储存最大的升序值
        if not nums:
            return 0
        for i in range(len(nums)-1):
            if nums[i]<nums[i+1]:
                tmp += 1
            else:
                tmp = 1
            if tmp>ret:
                ret = tmp
        return ret
            

```

## 67. 二进制求和

给定两个二进制字符串，返回他们的和（用二进制表示）。
输入为非空字符串且只包含数字 1 和 0。

示例 1:
输入: a = "11", b = "1"
输出: "100"

示例 2:
输入: a = "1010", b = "1011"
输出: "10101"

**解题思路：**
bin十进制转二进制，int将字符串转化为十进制

```Python
class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        return bin(int(a,2)+int(b,2))[2:]
```

## 125. 验证回文串
给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。

说明：本题中，我们将空字符串定义为有效的回文串。

示例 1:
输入: "A man, a plan, a canal: Panama"
输出: true

示例 2:
输入: "race a car"
输出: false

**解题思路：**
注意isalnum()的使用

```Python
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if not s:
            return True
        tmp = ''
        for i in s:
            if i.isalnum():
                tmp += i.lower()
        return tmp==tmp[::-1]
```

## 697. 数组的度
给定一个非空且只包含非负数的整数数组 nums, 数组的度的定义是指数组里任一元素出现频数的最大值。
你的任务是找到与 nums 拥有相同大小的度的最短连续子数组，返回其长度。

示例 1:
输入: [1, 2, 2, 3, 1]
输出: 2
解释: 
输入数组的度是2，因为元素1和2的出现频数最大，均为2.
连续子数组里面拥有相同度的有如下所示:
[1, 2, 2, 3, 1], [1, 2, 2, 3], [2, 2, 3, 1], [1, 2, 2], [2, 2, 3], [2, 2]
最短连续子数组[2, 2]的长度为2，所以返回2.

示例 2:
输入: [1,2,2,3,1,4,2]
输出: 6

注意:
nums.length 在1到50,000区间范围内。
nums[i] 是一个在0到49,999范围内的整数。

**解题思路：**
利用字典进行数值个数的统计，并针对出现次数最多的数值确定初始以及末尾对应的index，通过index相减获得对应的最短子序列

```Python
class Solution(object):
    def findShortestSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 对每个值的数量，起始index终止index进行统计
        adic = {}
        for k,v in enumerate(nums):
            if v in adic:
                adic[v][0] += 1
                adic[v][2] = k
            else:
                adic[v] = [1,k,-1]
        max_cnt = max(zip(*adic.values())[0])
        # 最大频数如果是1直接返回1
        if max_cnt==1:
            return 1
        # 确定最大频数并找到最短子序列长度
        min_len = float('Inf')
        for k,v in adic.items():
            if v[0]==max_cnt:
                len_tmp = v[2]-v[1]+1
                if len_tmp<min_len:
                    min_len = len_tmp
        return min_len
```


## 344. 反转字符串
编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 char[] 的形式给出。
不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。
你可以假设数组中的所有字符都是 ASCII 码表中的可打印字符。

示例 1：
输入：["h","e","l","l","o"]
输出：["o","l","l","e","h"]

示例 2：
输入：["H","a","n","n","a","h"]
输出：["h","a","n","n","a","H"]

**解题思路：**
根据python指针的特性，使用双指针，一个头一个尾，调换数值顺序

```Python
class Solution(object):
    def reverseString(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        i,j = 0,len(s)-1
        while i<j:
            s[i],s[j] = s[j],s[i]
            i += 1
            j -= 1
        return s
```


## 345. 反转字符串中的元音字母
编写一个函数，以字符串作为输入，反转该字符串中的元音字母。

示例 1:
输入: "hello"
输出: "holle"

示例 2:
输入: "leetcode"
输出: "leotcede"
说明:
元音字母不包含字母"y"。

**解题思路：**
其实跟反转字符串是一个套路，双指针即可，唯一需要注意的是需要对元音字母进行判断,注意转化成list

```Python
class Solution(object):
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """
        i,j=0,len(s)-1
        s_list = list(s)
        alp = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'] # 注意大写
        while i<j:
            if s_list[i] in alp:
                if s_list[j] in alp:
                    s_list[i],s_list[j] = s_list[j],s_list[i]
                    i += 1 # 记得换位置啊
                    j -= 1 # 记得换位置啊
                else:
                    j -= 1
            else:
                i += 1
        return ''.join(s_list)
```


## 717. 1比特与2比特字符

有两种特殊字符。第一种字符可以用一比特0来表示。第二种字符可以用两比特(10 或 11)来表示。
现给一个由若干比特组成的字符串。问最后一个字符是否必定为一个一比特字符。给定的字符串总是由0结束。

示例 1:
输入: 
bits = [1, 0, 0]
输出: True
解释: 
唯一的编码方式是一个两比特字符和一个一比特字符。所以最后一个字符是一比特字符。

示例 2:
输入: 
bits = [1, 1, 1, 0]
输出: False
解释: 
唯一的编码方式是两比特字符和两比特字符。所以最后一个字符不是一比特字符。

**解题思路：**
因为list只由0，10/11组成，观察特点，10/11都是以1开头，0是以0开头，因此正向遍历，遇到1跳两个格，因为只能在10/11中且长度为2，遇到0则跳一个格，因为0长度只能是1，最后看指针所指位置，如果是最后一次指针加了2那么说明不是以单个0结尾，否则就是以单个0结尾的

```Python
class Solution(object):
    def isOneBitCharacter(self, bits):
        """
        :type bits: List[int]
        :rtype: bool
        """
        i = 0
        while i<len(bits):
            if bits[i]==1:
                i += 2
                ret = False
            else:
                i += 1
                ret = True
        return ret
```


## 383. 赎金信
给定一个赎金信 (ransom) 字符串和一个杂志(magazine)字符串，判断第一个字符串ransom能不能由第二个字符串magazines里面的字符构成。如果可以构成，返回 true ；否则返回 false。
(题目说明：为了不暴露赎金信字迹，要从杂志上搜索各个需要的字母，组成单词来表达意思。)

注意：
你可以假设两个字符串均只含有小写字母。

canConstruct("a", "b") -> false
canConstruct("aa", "ab") -> false
canConstruct("aa", "aab") -> true

**解题思路:**
这个题的意思是：你要从magazine里面找字母，ransomNote中的每个字母都包含在magazine里面，且个数要小于magazine这个里面的个数

```Python
class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        adic1 = {};adic2={}
        for i in ransomNote:
            if i in adic1:
                adic1[i] += 1
            else:
                adic1[i] = 1
        for i in magazine:
            if i in adic2:
                adic2[i] += 1
            else:
                adic2[i] = 1
        for k,v in adic1.items():
            if k in adic2 and v<=adic2[k]:
                continue
            else:
                return False
        return True
        ```

## 387. 字符串中的第一个唯一字符
给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1。

案例:
s = "leetcode"
返回 0.

s = "loveleetcode",
返回 2.

**解题思路**
利用两个list，一个存储uniq字符，一个存储dup字符

```Python
class Solution(object):
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        uniq = []
        dup = []
        for i in s:
            if i in uniq:
                uniq.remove(i)
                if i in dup:
                    continue
                else:
                    dup.append(i)
            else:
                if i in dup:
                    continue
                else:
                    uniq.append(i)
        if uniq:
            return s.index(uniq[0])
        return -1
        ```

## 724. 寻找数组的中心索引
给定一个整数类型的数组 nums，请编写一个能够返回数组“中心索引”的方法。
我们是这样定义数组中心索引的：数组中心索引的左侧所有元素相加的和等于右侧所有元素相加的和。
如果数组不存在中心索引，那么我们应该返回 -1。如果数组有多个中心索引，那么我们应该返回最靠近左边的那一个。

示例 1:
输入: 
nums = [1, 7, 3, 6, 5, 6]
输出: 3
解释: 
索引3 (nums[3] = 6) 的左侧数之和(1 + 7 + 3 = 11)，与右侧数之和(5 + 6 = 11)相等。
同时, 3 也是第一个符合要求的中心索引。

示例 2:
输入: 
nums = [1, 2, 3]
输出: -1
解释: 
数组中不存在满足此条件的中心索引。
说明:

nums 的长度范围为 [0, 10000]。
任何一个 nums[i] 将会是一个范围在 [-1000, 1000]的整数。

**解题思路：**
先对list求和，之后对list进行遍历，每次遍历都对前半部分进行累加，并通过总和减去前半部分以及当前遍历值求解后半部分值，进行比较

```Python
class Solution(object):
    def pivotIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)<2:
            return -1
        sum_val = sum(nums)
        before = 0
        for k,v in enumerate(nums):
            if before==sum_val-before-v:
                return k
            before = before+v
        return -1
```


## ....
给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。
注意：答案中不可以包含重复的三元组。
例如, 给定数组 nums = [-1, 0, 1, 2, -1, -4]，

满足要求的三元组集合为：
[
  [-1, 0, 1],
  [-1, -1, 2]
]

```Python
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        '''
        三数之和的解题思路：
        首先对数组进行排序，之后对数字进行遍历，在遍历过程中如果是第一个或者当前的数字与之前的不相等进行判断，start和end分别是下一节点以及最后节点
        start<end进行while循环，如果两个游标以及当前数字之和为0，添加结果，同时两个游标同时更新，下一步跳过相同的
        如果不相等的话要么跟新左游标要么右游标
        '''
        ret = []# 记录结果
        nums = sorted(nums) # 一定要排序
        for i,v in enumerate(nums):
            start,end = i+1,len(nums)-1
            if i==0 or nums[i]>nums[i-1]:# 跳过相同的
                while start<end:
                    if nums[i]+nums[start]+nums[end]==0:
                        ret.append([nums[i],nums[start],nums[end]])
                        start += 1
                        end -= 1
                        while start<end and nums[start]==nums[start-1]:#跳过相同的
                            start += 1
                        while start<end and nums[end]==nums[end+1]:#跳过相同的
                            end -= 1
                    elif nums[i]+nums[start]+nums[end]>0:
                        end -= 1
                    else:
                        start += 1
        return ret
```




# Leetcode Hot100

## 两数之和
```Python
解题思路：
使用遍历，结合字典，记录每次target-val对应的值，如果遍历的值在其中，直接输出即可
```


## 2. 两数相加
```Python
输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
原因：342 + 465 = 807

解题思路：
遍历链表，得到数字进行int转换，进行运算，同时在构造新的链表
```


## 3. 无重复字符的最长子串
```Python
给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。

示例 1:

输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

解题思路：
初始化start和最长值为0，之后遍历整个数组，如果字符出现在之前的字符串中，start就+1，然后得到相应的长度，和最大值比较进行更新即可
```


## 4. 寻找两个有序数组的中位数
```Python
给定两个大小为 m 和 n 的有序数组 nums1 和 nums2。
请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。
你可以假设 nums1 和 nums2 不会同时为空。

示例 1:
nums1 = [1, 3]
nums2 = [2]
则中位数是 2.0

解题思路：
使用归并，对数组进行排序，得到新的数组之后再跟据长度求解相应的中位数
```


## 5. 最长回文子串
```Python
给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。

示例 1：
输入: "babad"
输出: "bab"
注意: "aba" 也是一个有效答案。

解题思路：
中心扩增法，遍历每个字符，使用辅助函数，使之以其idx-idx或者idx-idx+1进行查看回文，返回最大值，之后奇偶最大长度比较同时跟当前最大长度比较，进行字符串更新
```


## 11. 盛最多水的容器
```Python
给定 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0)。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
说明：你不能倾斜容器，且 n 的值至少为 2。

解题思路：
双指针，给定left=0，right=len(s)-1，maxarea=(right-left)*min(left,right)，之后while循环如果left<right，如果left低那么left+1否则right-1，不断计算更新最大面积
```


## 15.三数之和
```Python
给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。

注意：答案中不可以包含重复的三元组。

例如, 给定数组 nums = [-1, 0, 1, 2, -1, -4]，

满足要求的三元组集合为：
[
  [-1, 0, 1],
  [-1, -1, 2]
]

        '''
        三数之和的解题思路：
        首先对数组进行排序，之后对数字进行遍历，在遍历过程中如果是第一个或者当前的数字与之前的不相等进行判断，start和end分别是下一节点以及最后节点
        start<end进行while循环，如果两个游标以及当前数字之和为0，添加结果，同时两个游标同时更新，下一步跳过相同的
        如果不相等的话要么跟新左游标要么右游标
        '''
        ret = []# 记录结果
        nums = sorted(nums) # 一定要排序
        for i,v in enumerate(nums):
            start,end = i+1,len(nums)-1
            if i==0 or nums[i]>nums[i-1]:# 跳过相同的
                while start<end:
                    if nums[i]+nums[start]+nums[end]==0:
                        ret.append([nums[i],nums[start],nums[end]])
                        start += 1
                        end -= 1
                        while start<end and nums[start]==nums[start-1]:#跳过相同的
                            start += 1
                        while start<end and nums[end]==nums[end+1]:#跳过相同的
                            end -= 1
                    elif nums[i]+nums[start]+nums[end]>0:
                        end -= 1
                    else:
                        start += 1
        return ret
```


## 17. 电话号码的字母组合
```Python
给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。
给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

解题思路：
对于每个数字建立字典，然后对每个数字的多个字符进行遍历，将之前得到的字符串与之相加，并重新赋值，不断这样循环
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if not digits:
            return []
        adic = {'2':'abc', '3':'def', '4':'ghi', '5':'jkl', '6':'mno', \
                '7':'pqrs', '8':'tuv','9':'wxyz'}
        ret = ['']
        for i in digits:
            tmp = []
            for j in ret: # 依次循环上次得到的list
                for k in list(adic[i]): # 之前元素加上新的元素
                    tmp.append(j + k)
            ret = tmp # 更新列表
        return ret
```


## 19. 删除链表的倒数第N个节点
```Python
给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点。
示例：
给定一个链表: 1->2->3->4->5, 和 n = 2.
当删除了倒数第二个节点后，链表变为 1->2->3->5.

解题思路：
设置快慢指针，快指针先走n，然后慢指针快指针同时走，当快指针next是空时，将慢指针next跳一个，返回结果
```


## 20. 有效的括号
```Python
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
有效字符串需满足：
左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
注意空字符串可被认为是有效字符串。

解题思路：
使用栈实现，对字符串进行遍历，如果是左括号，那么需要压栈，如果是右括号的话就弹栈比较，如果不匹配就false，如果最后栈不空false否则true
```


## 22. 括号生成
```Python
给出 n 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且有效的括号组合。

例如，给出 n = 3，生成结果为：
[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]

解题思路：
使用动态规划方法，对于当前括号的值可以遍历上一次括号中的所有值，并在每个空位都进行插入，返回去重后的结果
class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        if n==0:
            return []
        elif n==1:
            return ['()']
        else:
            i=2
            before=['()']
            while i<=n:
                after = []
                for st in before:
                    for k in range(len(st)):
                        after.append(st[:k]+'()'+st[k:])
                after = list(set(after))
                before = after
                i += 1
            return after
```


## 23. 合并K个排序链表
```Python
合并 k 个排序链表，返回合并后的排序链表。请分析和描述算法的复杂度。

解题思路：
将所有链表中的值合并到一个list中进行排序，然后排好序之后，重新组织进行输出

class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        self.nodes = []
        head = point = ListNode(0)
        for l in lists:
            while l:
                self.nodes.append(l.val)
                l = l.next
        for x in sorted(self.nodes):
            point.next = ListNode(x)
            point = point.next
        return head.next

```


## 46. 全排列
```Python
解题思路：
这里不使用回溯法，相反的使用一个较为简单的方式，即动态规划，每次对列表进行遍历，然后，对之前的列表在不同位置进行插入当前元素，之后更新之前元素，不断轮回即可

class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums)<2:
            return [nums]
        else:
            pre = [[nums[0]]]
            for i in nums[1:]:
                after = []
                for j in pre:
                    for k in range(0,len(j)+1):
                        after.append(j[:k]+[i]+j[k:])
                pre = after
            return after
```


## 32. 最长有效括号
```Python
解题思路：
使用辅助函数判断当前的括号是否为有效括号，然后选定偶数括号，一一进行判断，找到最长的即可
```


## 33. 搜索旋转排序数组
```Python
假设按照升序排序的数组在预先未知的某个点上进行了旋转。
( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。
你可以假设数组中不存在重复的元素。
你的算法时间复杂度必须是 O(log n) 级别。

解决思路：
直接二分进行查找，只不过判断的时候需要分情况进行考虑
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        '''解题思路：
        直接使用二分查找，需要判断边界，进而重新界定left和right
        '''
        if len(nums)==0:
            return -1
        left,right = 0,len(nums)-1
        while left<right:
            mid = left+(right-left)//2
            if nums[mid]==target:
                return mid
            elif nums[mid]>=nums[left]:
                if nums[left]<=target<nums[mid]:
                    right = mid-1
                else:
                    left = mid+1
            else:
                if nums[mid]<target<=nums[right]:
                    left = mid+1
                else:
                    right = mid-1
        return left if nums[left]==target else -1
```


## 34. 在排序数组中查找元素的第一个和最后一个位置
```Python
解题思路：
找到相应的头以及尾部位置，然后进行range的求取

class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        def first(nums, target):
            l,r = 0,len(nums)-1
            while l<r:
                m = l+(r-l)/2
                if nums[m]>target:
                    r = m-1
                elif nums[m]<target:
                    l = m+1
                else:
                    if (m==0) or (m>0 and nums[m-1]!=nums[m]):
                        return m
                    else:
                        r = m-1
            return l
        def last(nums, target):
            l,r = 0,len(nums)-1
            while l<r:
                m = l+(r-l)/2
                if nums[m]>target:
                    r = m-1
                elif nums[m]<target:
                    l = m+1
                else:
                    if (m==len(nums)-1) or (m<len(nums)-1 and nums[m]!=nums[m+1]):
                        return m
                    else:
                        l = m+1
            return r

        if len(nums)==0:
            return [-1,-1]
        s = first(nums, target)
        e = last(nums, target)
        return [s,e] if nums[s]==target else [-1,-1]

```


## 48. 旋转图像
```Python
解题思路：
只要把数组先进行转置，在对每行求逆即可
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        row,col = len(matrix),len(matrix[0])
        for i in range(row):
            for j in range(i+1,col):
                matrix[i][j],matrix[j][i] = matrix[j][i],matrix[i][j]
        for i in range(row):
            matrix[i] = matrix[i][::-1]
        return matrix
```


## 49. 字母异位词分组
```Python
给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

示例:

输入: ["eat", "tea", "tan", "ate", "nat", "bat"],
输出:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]

解题思路：
使用sorted tuple以及字典进行记录，最后返回values
```


## 50. Pow(x, n)
```Python
使用递归，注意对n的符号进行判断
```


## 55. 跳跃游戏
```Python
给定一个非负整数数组，你最初位于数组的第一个位置。
数组中的每个元素代表你在该位置可以跳跃的最大长度。
判断你是否能够到达最后一个位置。

示例 1:
输入: [2,3,1,1,4]
输出: true
解释: 从位置 0 到 1 跳 1 步, 然后跳 3 步到达最后一个位置。

解题思路：
使用贪心算法+动态规划算法，从后向前设置start、end，如果start+nums[start]就更新end，同时每次start都更新，最后查看是否end为0，如果为0说明可以从0开始跳到最后一个位置
```


## 62. 不同路径
```Python
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

问总共有多少条不同的路径？

解题思路：
动态规划，dp[i][j] = dp[i-1][j]+dp[i][j-1]
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        dp = [[1]*n] + [[1]+[0]*(n-1) for _ in range(m-1)]
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = dp[i-1][j]+dp[i][j-1]
        return dp[-1][-1]
```


## 73. 矩阵置零
```Python
给定一个 m x n 的矩阵，如果一个元素为 0，则将其所在行和列的所有元素都设为 0。请使用原地算法。

解题思路：
对0进行记录，使用set进行记录，在此遍历元素，如果行列在里面就直接置为0
class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        row,col = len(matrix),len(matrix[0])
        rows=set()
        cols=set()
        for i in range(row):
            for j in range(col):
                if matrix[i][j]==0:
                    rows.add(i)
                    cols.add(j)
        for i in range(row):
            for j in range(col):
                if i in rows or j in cols:
                    matrix[i][j] = 0
```


## 78. 子集
```Python
给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。

说明：解集不能包含重复的子集。

示例:

输入: nums = [1,2,3]
输出:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]

解题思路：
动态规划，每增加一个元素其实就是原来list基础上再加上对所有list元素+新的值 产生的list
```


## 79. 单词搜索
```Python
解题思路：
跟之前的路径搜索一样，回溯，设定state以及路径长度，进行拟合
```


## 91. 解码方法
```Python
一条包含字母 A-Z 的消息通过以下方式进行了编码：

'A' -> 1
'B' -> 2
...
'Z' -> 26
给定一个只包含数字的非空字符串，请计算解码方法的总数。

示例 1:

输入: "12"
输出: 2
解释: 它可以解码为 "AB"（1 2）或者 "L"（12）。
示例 2:

输入: "226"
输出: 3
解释: 它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。

解题思路：
动态规划，起始点是1/看本身是否为0来确定，然后不断更新即可，如果两个字母成立就加before如果一个字母成立就加after
```


## 94. 二叉树的中序遍历
```Python
解题思路：
递归+非递归
非递归，自己写的栈，再加一个cur指针，while栈和指针不空，while指针不空，就不断左节点遍历，然后弹栈输出值
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
#       递归
#        if not root:
#            return []
#        ret = []
#        ret += self.inorderTraversal(root.left)
#        ret.append(root.val)
#        ret += self.inorderTraversal(root.right)
#        return ret
#       非递归
        ret = []
        stack = []
        p = root
        while p or stack:
            while p:
                stack.append(p)
                p = p.left
            tmp = stack.pop()
            ret.append(tmp.val)
            p = tmp.right
        return ret
```


## 98. 验证二叉搜索树
```Python
给定一个二叉树，判断其是否是一个有效的二叉搜索树。

假设一个二叉搜索树具有如下特征：

节点的左子树只包含小于当前节点的数。
节点的右子树只包含大于当前节点的数。
所有左子树和右子树自身必须也是二叉搜索树

解题思路：
对于二叉搜索树来说，中序遍历是有序的，可以通过中序遍历结果查看是否有序进行判断
```


## 102. 二叉树的层次遍历
```Python
给定一个二叉树，返回其按层次遍历的节点值。 （即逐层地，从左到右访问所有节点）。

例如:
给定二叉树: [3,9,20,null,null,15,7],

[
  [3],
  [9,20],
  [15,7]
]

解题思路：
参见剑指offer
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        else:
            stack = [root]
            ret = []
            while stack:
                tmp = []
                length = len(stack)
                for _ in range(length):
                    cur = stack.pop(0)
                    if cur.left:
                        stack.append(cur.left)
                    if cur.right:
                        stack.append(cur.right)
                    tmp.append(cur.val)
                ret.append(tmp)
            return ret
```


## 103. 二叉树的锯齿形层次遍历
```Python
解题思路：
给定一个二叉树，返回其节点值的锯齿形层次遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。

例如：
给定二叉树 [3,9,20,null,null,15,7],

在层次遍历的基础上加一个flag判断行数即可
```


## 105. 从前序与中序遍历序列构造二叉树
```Python
解题思路：
跟剑指offer思路一致
```


## 116. 填充每个节点的下一个右侧节点指针
```Python
解题思路：
依旧是层次遍历的变体，只需要记录每一层的节点数，同时再加上next指针即可，同时不断进行每一层的更新
```


## 127. 单词接龙
```Python
给定两个单词（beginWord 和 endWord）和一个字典，找到从 beginWord 到 endWord 的最短转换序列的长度。转换需遵循如下规则：

每次转换只能改变一个字母。
转换过程中的中间单词必须是字典中的单词。
说明:

如果不存在这样的转换序列，返回 0。
所有单词具有相同的长度。
所有单词只由小写字母组成。
字典中不存在重复的单词。
你可以假设 beginWord 和 endWord 是非空的，且二者不相同。
示例 1:

输入:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

输出: 5
解释: 一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog",
     返回它的长度 5。

解题思路：
使用广度优先搜索，找最短路径，每一层是上一层可以变化的字符串同时要在字符串list中，不断遍历，找到短值
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        if not beginWord or not endWord or not wordList:
            return 0
        else:
            set_words = set(wordList)
            stack = [beginWord]
            nextstack = []
            length = 1
            while stack:
                for word in stack:
                    if word==endWord:
                        return length
                    else:
                        for i in range(len(word)):
                            for j in 'abcdefghijklmnopqrstuvwxyz':
                                newword = word[:i]+j+word[(i+1):]
                                if newword in set_words:
                                    set_words.remove(newword)
                                    nextstack.append(newword)
                length += 1
                stack = nextstack
                nextstack = []
            return 0
```


## 130. 被围绕的区域
```Python
给定一个二维的矩阵，包含 'X' 和 'O'（字母 O）。
找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。

示例:
X X X X
X O O X
X X O X
X O X X
运行你的函数后，矩阵变为：
X X X X
X X X X
X X X X
X O X X

解题思路：
转化问题，将问题转化为相应的非包裹问题，从矩阵四条边出发遍历(因为从四条边出发的一定是包不住的)，使用dfs+递归找到所有最深的包不上的路径，将这些路径标记为#，然后对所有元素进行遍历，如果是#则转化为O，如果是O，转化为X
```


## 131. 分割回文串
```Python
给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。

返回 s 所有可能的分割方案。

示例:

输入: "aab"
输出:
[
  ["aa","b"],
  ["a","a","b"]
]

解题思路：
回溯法+递归，设置help函数，给定start以及存储结果的list，对于list每个元素进行遍历，作为end，如果start到end是回文数，加入ret同时递归查找，终止条件是start>=字符串长度，则找到一条，加入真正结果list中
class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        if not s:
            return []
        else:
            result = []
            def getstr(string, start, ret=[]):
                if start>=len(string):
                    return result.append(ret)
                for end in range(start+1,len(string)+1):
                    if string[start:end]==string[start:end][::-1]:
                        getstr(string, end, ret+[string[start:end]])
            getstr(s, 0, ret=[])
            return result       
```


## 134. 加油站
```Python
在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。

你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。

如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。

说明: 

如果题目有解，该答案即为唯一答案。
输入数组均为非空数组，且长度相同。
输入数组中的元素均为非负数。

解题思路：
遍历，如果cur<0就更新，否则一直下去
class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        cur,total=0,0
        start = 0
        for i in range(len(gas)):
            cur += gas[i]-cost[i]
            total += gas[i]-cost[i]
            if cur<0:
                start = i+1;cur = 0
        return -1 if total<0 else start

```


## 138. 复制带随机指针的链表
```Python
给定一个链表，每个节点包含一个额外增加的随机指针，该指针可以指向链表中的任何节点或空节点。
要求返回这个链表的深拷贝。 

解题思路：
使用三步法进行链表的复制，但是注意如果random节点存在的时候才能copy
```


## 139. 单词拆分
```Python
给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。

说明：

拆分时可以重复使用字典中的单词。
你可以假设字典中没有重复的单词。
示例 1：

输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。

解题思路：
使用动态规划，start从头开始遍历，end从start+1开始遍历，初始化一个flag list存储当前单词状态，如果s[start:end]在worddic中，就对该下标end对应的位置设置1，然后从值为1的位置向后查看，如果左后为1返回true
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        flag = [1]+[0]*len(s)
        for start in range(len(s)):
            if flag[start]==1:
                for end in range(start+1,len(s)+1):
                    if s[start:end] in wordDict:
                        flag[end] = 1
        return flag[-1]
```


## 150. 逆波兰表达式求值
```Python
根据逆波兰表示法，求表达式的值。

有效的运算符包括 +, -, *, / 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。

说明：

整数除法只保留整数部分。
给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。
示例 1：

输入: ["2", "1", "+", "3", "*"]
输出: 9
解释: ((2 + 1) * 3) = 9

解题思路:
使用栈，只要遇到了符号，就直接进行运算并把值压入，知道输出在后一个值
```


## 152. 乘积最大子序列
```Python
给定一个整数数组 nums ，找出一个序列中乘积最大的连续子序列（该序列至少包含一个数）。

示例 1:

输入: [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。

解题思路：
其实跟最大子序列和一样的，区别在于乘法可能正负颠倒，因此要同时记录最大最小值，如果当前值是负数，那么点到最大最小值
```


## 162. 寻找峰值
```Python
峰值元素是指其值大于左右相邻值的元素。
给定一个输入数组 nums，其中 nums[i] ≠ nums[i+1]，找到峰值元素并返回其索引。
数组可能包含多个峰值，在这种情况下，返回任何一个峰值所在位置即可。
你可以假设 nums[-1] = nums[n] = -∞。

示例 1:
输入: nums = [1,2,3,1]
输出: 2
解释: 3 是峰值元素，你的函数应该返回其索引 2。

解题思路：
看到复杂度使用二分查找+递归，终止条件左右相等，之后如果点在上升序列中就递归右半边，否则递归左半边
```


## 163. 缺失的区间
```Python
给定一个排序的整数数组 nums ，其中元素的范围在 闭区间 [lower, upper] 当中，返回不包含在数组中的缺失区间。

示例：

输入: nums = [0, 1, 3, 50, 75], lower = 0 和 upper = 99,
输出: ["2", "4->49", "51->74", "76->99"]

解题思路：
主要是将最小值最大值插入到list中，在进行遍历即可
class Solution(object):
    def findMissingRanges(self, nums, lower, upper):
        """
        :type nums: List[int]
        :type lower: int
        :type upper: int
        :rtype: List[str]
        """
        if not nums or lower<nums[0]:
            nums = [lower-1] + nums
        if not nums or upper>nums[-1]:
            nums.append(upper+1)
        ret = []
        for i in range(len(nums)-1):
            if nums[i+1]-nums[i]==2:
                ret.append(str(nums[i]+1))
            if nums[i+1]-nums[i]>2:
                ret.append(str(nums[i]+1)+'->'+str(nums[i+1]-1))
        return ret
```


## 179. 最大数
```Python
给定一组非负整数，重新排列它们的顺序使之组成一个最大的整数。

示例 1:

输入: [10,2]
输出: 210
示例 2:

输入: [3,30,34,5,9]
输出: 9534330

解题思路：
转化为字符转之后直接使用比较器进行比较

class Solution(object):
    def largestNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        nums = [str(i) for i in nums]
        nums = sorted(nums, cmp=lambda y,x:int(x+y)-int(y+x))
        return str(int(''.join(nums)))
```


## 200. 岛屿数量
```Python
给定一个由 '1'（陆地）和 '0'（水）组成的的二维网格，计算岛屿的数量。一个岛被水包围，并且它是通过水平方向或垂直方向上相邻的陆地连接而成的。你可以假设网格的四个边均被水包围。

示例 1:
输入:
11110
11010
11000
00000
输出: 1

解题思路：
使用广度遍历，遍历所有点，以每个点为起始点进行bfs，然后记录个数，最后返回个数
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if not grid:
            return 0
        row = len(grid)
        col = len(grid[0])
        if row<=0 or col<=0:
            return 0
        state = [[0 for _ in range(col)] for _ in range(row)]
        ret = 0
        for i in range(row):
            for j in range(col):
                if state[i][j]==0 and grid[i][j]=='1':
                    ret += 1
                    state[i][j] = 1
                    stack = [[i,j]]
                    while stack:
                        cur = stack.pop(0)
                        for k in [[1,0],[-1,0],[0,1],[0,-1]]:
                            newi = cur[0]+k[0];newj = cur[1]+k[1];
                            if newi>=0 and newi<row and newj>=0 and newj<col and state[newi][newj]==0 and grid[newi][newj]=='1':
                                stack.append([newi,newj])
                                state[newi][newj] = 1
                    # print state
        return ret
```


## 207. 课程表
```Python
现在你总共有 n 门课需要选，记为 0 到 n-1。

在选修某些课程之前需要一些先修课程。 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示他们: [0,1]

给定课程总量以及它们的先决条件，判断是否可能完成所有课程的学习？

示例 1:

输入: 2, [[1,0]] 
输出: true
解释: 总共有 2 门课程。学习课程 1 之前，你需要完成课程 0。所以这是可能的。

解题思路：
使用拓扑排序的思想，需要准备入度list，邻接表，然后对于入度为0就加入，如果遍历完了之后值和课程数相等就返回true

class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        # 拓扑排序
        # 需要关于度的list/需要接临表
        innode = [0 for i in range(numCourses)]
        nextdic = {}
        for i in range(numCourses):
            nextdic[i] = []
        stack = []
        for i,j in prerequisites:
            innode[i] += 1
            nextdic[j] += [i]
        for k,v in enumerate(innode):
            if v==0:
                stack.append(k)
        ret = 0
        while stack:
            cur = stack.pop(0)
            ret += 1
            # print nextdic,cur
            for i in nextdic[cur]: # 记得对每一值都要初始化空list
                innode[i] -= 1
                if innode[i]==0:
                    stack.append(i)
        return numCourses==ret
```


## 210. 课程表 II
```Python
现在你总共有 n 门课需要选，记为 0 到 n-1。
在选修某些课程之前需要一些先修课程。 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示他们: [0,1]
给定课程总量以及它们的先决条件，返回你为了学完所有课程所安排的学习顺序。
可能会有多个正确的顺序，你只要返回一种就可以了。如果不可能完成所有课程，返回一个空数组。

示例 1:
输入: 2, [[1,0]] 
输出: [0,1]
解释: 总共有 2 门课程。要学习课程 1，你需要先完成课程 0。因此，正确的课程顺序为 [0,1] 。

解题思路：
依旧是拓扑排序，设定邻接表以及入度list，然后进行遍历，同时不断加入相应的弹出节点，如果最后有效的话就就返回顺序的list否则返回[]
class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        # 拓扑排序
        # 构造邻接字典，入度list
        innode = [0 for _ in range(numCourses)]
        nextdic = {}
        for i in range(numCourses):
            nextdic[i] = []
        for i,j in prerequisites:
            innode[i] += 1
            nextdic[j] += [i]
        # 节点为0的node
        stack = []
        for k,v in enumerate(innode):
            if v==0:
                stack.append(k)
        
        # 循环
        ret = 0
        retlist = []
        while stack:
            cur = stack.pop(0)
            ret += 1
            retlist.append(cur)
            # 去除连接
            for i in nextdic[cur]:
                innode[i] -= 1
                if innode[i]==0:
                    stack.append(i)
        return retlist if ret==numCourses else []
```






# 排序算法

## 1.归并排序
**解题思路：**
如果序列长度大于1可以对序列进行拆分，排序，再进行合并
当拆分为只有单个元素的list之后默认就是排好序的
这样每次迭代的子部分相当于排好序的list，只需对两个排好序的list进行merge即可
将这种思想进行扩展即可
[归并排序思路](https://www.jianshu.com/p/3ad5373465fd)

```Python
def mergeSort(nums):
	if len(nums)==1:
		return nums
	else:
		mid = len(nums)/2
		left = mergesort(nums[:mid])
		right = mergesort(nums[mid:])
		return merge(left, right)
def merge(left, right):
	ret = []
	i,j = 0,0
	while i<len(left) and j<len(right):
		if left[i]<=right[j]:
			ret.append(left[i])
			i += 1
		else:
			ret.append(right[j])
			j += 1
	if i==len(left):
		ret += right[j:]
	else:
		ret += left[i:]
	return ret

test = [7,3,1,4,5]
print 'list before sort: %s'%test
print 'list after sort: %s'%mergeSort(test)
```

## 2.桶排序

**解题思路:**
桶排序是一种计数排序，也就是说，建立最小值到最大值之间的list，
计算每个数字在nums中出现的次数，填写在在相应list的位置，
最后遍历整个list输出值不为出代表的nums中的数字即可
(不适用于范围太大元素个数少的list)
[桶排序思路](https://www.cnblogs.com/king-ding/p/bucketsort.html)
```Python
def bucketSort(nums):
	max_num = max(nums)
	min_num = min(nums)
	init_list = [0]*(max_num-min_num+1)
	ret_list = []
	for i in nums:
		init_list[i-min_num] += 1
	for k,v in enumerate(init_list):
		if v!=0:
			ret_list += [k+min_num]*v
	return ret_list

test = [7,3,1,4,5]
print 'list before sort: %s'%test
print 'list after sort: %s'%bucketSort(test)
```

## 3.快速排序

**解题思路:**
选择第一个位置作为一个标定，之后遍历后面的list，找出比这个数大的，放在右边，
找出比这个数小的放在左边，进一步迭代排序左边和右边子list，即可获得排序后的结果
(有一点二分的思想)
[快排思路1](http://yshblog.com/blog/170)
[快排思路2](https://blog.csdn.net/razor87/article/details/71155518)
[快排递归非递归](https://blog.csdn.net/qq_34178562/article/details/79903871)

```Python
def quickSort(nums):
	if len(nums)<=1:
		return nums
	else:
		left = [i for i in nums[1:] if i<=nums[0]]
		right = [i for i in nums[1:] if i>nums[0]]
		ret = quickSort(left)+[nums[0]]+quickSort(right)
		return ret

def quickSort(nums):

test = [7,3,1,4,5]
print 'list before sort: %s'%test
print 'list after sort: %s'%bucketSort(test)
```

## 4.基数排序

**解题思路:**
基数排序主要是这样实现的，首先找出所有数字的最大值，通过最大值得到最大数的位数，
建立一个空值的list，分别代表0-9十个数。然后通过对nums中的元素的个位数进行提取，进而排序，
放到相应的空值list的位置处，之后将遍历完的数组进行合并，作为一次排序的结果，以此类推，
直到所有位数都排完，即可得到排序后的数组
[基数排序思路](https://baike.baidu.com/item/%E5%9F%BA%E6%95%B0%E6%8E%92%E5%BA%8F)
```Python
def jishuSort(nums):
	k = int(math.ceil(math.log(max(nums), 10)))
	alist = [[] for _ in range(10)]
	for i in range(k):
		for v in nums:
			alist[(v/10**i)%10].append(v)
		nums = []
		for j in alist:
			nums.extend(j)
		alist = [[] for _ in range(10)]
	return nums

test = [10,30,123,34,22,11]
print 'list before sort: %s'%test
print 'list after sort: %s'%jishuSort(test)
```


## 5.二叉树排序
**解题思路：**
二叉树排序主要有几个步骤，构建节点，通过插值构建二叉树，对构建好的二叉树进行遍历
[二叉树排序思路](https://blog.csdn.net/Rex_WUST/article/details/83274507)
```Python
class Tree(object):
	def __init__(self, value):
		self.value = value
		self.left = None
		self.right = None
	def leftInsert(self, value):
		self.left = Tree(value)
		return self.left
	def rightInsert(self, value):
		self.right = Tree(value)
		return self.right
	def show(self):
		return self.value

def insertValue(node, value):
	if node.value>value:
		if node.left:
			insertValue(node.left, value)
		else:
			node.leftInsert(value)
	else:
		if node.right:
			insertValue(node.right, value)
		else:
			node.rightInsert(value)

def leftOrder(node):
	if node.value:
		if node.left:
			leftOrder(node.left)
		alist.append(node.show()) # alist是全局变量
		if node.right:
			leftOrder(node.right)

alist = []
nums = [2,3,1,6,4]
root = Tree(nums[0])
tree = root
for i in nums[1:]:
	insertValue(tree, i)
leftOrder(tree)

print 'list before sort: %s'%nums
print 'list after sort: %s'%alist
```


## 6. 堆排序
[实例代码](https://zhuanlan.zhihu.com/p/58221959)
[参考](https://www.jianshu.com/p/d174f1862601)
```Python
def help(arr, s, e):
    N = e-1
    while s*2<=N:
        j = s*2
        if j<N and arr[j]<arr[j+1]:
            j += 1
        if arr[s]<arr[j]:
            arr[s],arr[j] = arr[j],arr[s]
            s = j
        else:
            break

def heapsort(arr):
    N = len(arr)-1
    for i in range(N//2, 0, -1):
        help(arr, i, len(arr))
    while N>1:
        arr[1],arr[N] = arr[N],arr[1]
        help(arr, 1, N)
        N -= 1
    return arr[1:]

heapsort([-1, 2,1,7,6,3,4,5]) # -1是用来占位的

```


## 7. topK

基于快排
[参考](https://blog.csdn.net/lanchunhui/article/details/50960895)

```Python
def partition(arr):
    m = arr[0]
    l = [i for i in arr[1:] if i<=m]
    h = [i for i in arr[1:] if i>m]
    return l,m,h
def select(arr, k):
    l,m,h = partition(arr)
    if len(l)==k-1:
        return m
    elif len(l)<k-1:
        return select(h, k-len(l)-1)
    else:
        return select(l, k)
select([2,1,7,6,3,4,5], 5)
```


# KMP字符搜索

[原理参考](http://www.ruanyifeng.com/blog/2013/05/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm.html)

```Python
def word_table(p):
    pre = set()
    post = set()
    ret = [0]
    for i in range(1,len(p)):
        pre.add(p[:i])
        post = set([p[j:i+1] for j in range(1,i+1)])
        ret.append(max(map(len, pre & post)+[0]))
    return ret
def KMP(s, p):
    N = len(s)
    M = len(p)
    cur = 0
    table = word_table(p)
    while cur <=N-M:
        for i in range(M):
            if s[cur+i]!=p[i]:
                cur += max(1, i-table[i-1])
                break
        else:
            return True
    return False 
print KMP("BBC ABCDAB ABCDABCDABDE", "ABCDABD")
print word_table('ABCDABR')
```


# 深度优先与广度优先

## 1. BFS
```Python
#! usr/bin/python
# -*- coding: UTF-8 -*-

graph = {
    'A':['B', 'C'],
    'B':['A', 'C', 'D'],
    'C':['A','B','D','E'],
    'D':['B','C','E','F'],
    'E':['C','D'],
    'F':['D']
    }
def BFS(graph, s):
    queue = []
    queue.append(s)
    seen = set()
    seen.add(s)
    parent={s:None} # 记录父节点
    while queue:
        cur = queue.pop(0)
        nodes = graph[cur]
        for i in nodes:
            if i not in seen:
                queue.append(i)
                seen.add(i)
                parent[i] = cur # 记录父节点
        print cur
    return parent
parent = BFS(graph,'E')

print ''
v = 'B' # 终点，最短路径
while v:
    print v
    v = parent[v]
```

## 2. DFS
```Python
def DFS(graph, s):
    stack = []
    stack.append(s)
    seen = set()
    seen.add(s)
    while stack:
        cur = stack.pop()
        nodes = graph[cur]
        for i in nodes:
            if i not in seen:
                stack.append(i)
                seen.add(i)
        print cur
# DFS(graph,'E')
```


# 二叉树

## 1. 前序遍历

```Python
def pre(root):
    if not root:
        return root
    else:
        stack = [root]
        cur = root
        while stack:
            node = stack.pop()
            print node.val
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
```

## 2. 中序遍历

```Python
def mid(root):
    if not root:
        return None
    else:
        cur = root
        stack = [root]
        while stack:
            while cur.left:
                stack.append(cur.left)
                cur = cur.left
            node = stack.pop()
            print node.val
            if node.right:
                stack.append(node.right)
                cur = node.right
```

## 3. 后序遍历

```Python
def post(root):
    if not root:
        return None
    else:
        stack1 = [root]
        stack2 = []
        while stack1:
            node = stack1.pop()
            stack2.append(node)
            if node.left:
                stack1.append(node.left)
            if node.right:
                stack1.append(node.right)
        while stack2:
            print stack.pop().val
```













<hr />










