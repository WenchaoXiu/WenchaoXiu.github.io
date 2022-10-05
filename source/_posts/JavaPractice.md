---
title: Java基本练习题
tags: []
date: 2018-09-1 17:09:27
permalink:
categories: Java
description: 这部分主要通过一些练习题对Java进行自学
image:
---
<p class="description"></p>


<!-- more -->

# 基本习题
## 练习题1
```java
import java.util.ArrayList;

public class Prac1 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		/*有一对兔子，从出生后第3个月起每个月都生一对兔子，小兔子长到第三个月后每个月又生一
		对兔子，假如兔子都不死，问每个月的兔子总数为多少？*/
		
		System.out.println(rabbitNumber(7));
	}
	public static int rabbitNumber(int n) {
		ArrayList<Integer> arr = new ArrayList<Integer>();
		arr.add(1);
		arr.add(1);
		if (n<=2) {
			return 1;
		}
		else {
			for (int i=2; i<n; i++) {
				arr.add(arr.get(i-1)+arr.get(i-2));//获得元素
			}
		}
		return arr.get(arr.size()-1);
	}

}
```

## 练习题2
```java
import java.util.ArrayList;

public class Prac2 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		// 判断101-200之间有多少个素数，并输出所有素数。 
		int count = 0;
		ArrayList<Integer> arr = new ArrayList<Integer>();
		for (int i=101; i<201; i++) {
			if (isPrime(i)) {
				count ++;
				arr.add(i);
			}
		}
		System.out.println("101-200之间的质数是:"+arr);
		System.out.println("101-200之间的质数个数是:"+count);
	}
	
	public static boolean isPrime(int n) {
		for (int i=2;i<n;i++) {
			if (n%i==0)
				return false;
		}
		return true;
	}
}
```

## 练习题3
```java
import java.util.ArrayList;

public class Prac3 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		/* 打印出所有的"水仙花数"，所谓"水仙花数"
		 * 是指一个三位数，其各位数字立方和等于该数本身。
		 */
		ArrayList<Integer> arr = new ArrayList<Integer>();
		for (int i=100; i<1000; i++) {
			if (isSpecial(i))
				arr.add(i);
		}
		System.out.println("水仙花数是: "+arr);

	}
	public static boolean isSpecial(int i) {
		int n100,n10,n1;
		n100 = i/100;
		n10 = i%100/10;
		n1 = i%100%10;
		int tmp = (int) (Math.pow(n1, 3)+Math.pow(n10, 3)+Math.pow(n100,3));
		if (tmp==i) {
			return true;
		}
		return false;
	}
}
```

## 练习题4
```java
import java.util.*;

public class Prac4 {
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		//将一个正整数分解质因数。例如：输入90,打印出90=2*3*3*5。 
		System.out.println("请输入正数:");
		Scanner in = new Scanner(System.in);
		int n = in.nextInt();
		String out = String.format("%d=", n);
		ArrayList<Integer> tmp = fisrtFactor(n);
		ArrayList<Integer> factor = new ArrayList<Integer>();
		if (tmp.size()==0)
			System.out.println(n+" 是质数");
		while (tmp.size()!=0) {
			factor.add(tmp.get(0));
			int last = tmp.get(1);
			tmp = fisrtFactor(tmp.get(1));
			if (tmp.size()==0) {
				factor.add(last);
			}
//			System.out.println(tmp);
		}
		for (int i=0;i<factor.size();i++) {
			if (i==0)
				out += String.format("%d", factor.get(i));
			else
				out += String.format("*%d", factor.get(i));
		}
		System.out.println(out);
	}
	public static ArrayList<Integer> fisrtFactor(int n) {
		ArrayList<Integer> arr = new ArrayList<Integer>();
		for(int i=2; i<n; i++) {
			if (n%i==0) {
				arr.add(i);
				arr.add(n/i);
				break;
			}
		}
		return arr;
	}
}
```

## 练习题5
```java

public class Prac5 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		/*利用条件运算符的嵌套来完成此题：
		 * 学习成绩>=90分的同学用A表示，
		 * 60-89分之间的用B表示，60分以下C
		 */
		System.out.println(Grade(95));
		System.out.println(Grade(87));
		System.out.println(Grade(57));
	}
	public static String Grade(int n) {
		String grade;
		if (n>=90) {
			grade = "A";
		}
		else if(n>=60){
			grade = "B";
		}else {
			grade = "C";
		}
		return grade;
	}
}
```

## 练习题6
```java
import java.util.*;

public class Prac6 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner in = new Scanner(System.in);
		System.out.println("请输入两个正数:");
		int a = in.nextInt();
		int b = in.nextInt();
		System.out.println("最小公约数"+zdgys(a,b));
		System.out.println("最小公倍数"+zxgbs(a,b));
	}
	
	public static int zdgys(int a, int b) {
		int m;
		int n;
		while (b!=0) {
			m = a/b;
			n = a%b;
			a = b;
			b = n;
		}
		return a;
	}
	
	public static int zxgbs(int a, int b) {
		int yueshu = zdgys(a, b);
		return a/yueshu*b;
	}
}
```

## 练习题7
```java
import java.util.Scanner;

public class Prac7 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		/*题目：输入一行字符，分别统计出其中英文
		 * 字母、空格、数字和其它字符的个数。 
		 */
		Scanner in = new Scanner(System.in);
		String str = in.nextLine();
		statistic(str);
	}
	
	public static void statistic(String str) {
		int letterCount = 0,spaceCount = 0;
		int numberCount = 0,otherCount = 0;
		char[] chrArr = str.toCharArray();
		for (char i : chrArr) {
			if (Character.isLetter(i)) {
				letterCount++;
			}
			else if (Character.isSpaceChar(i)) {
				spaceCount++;
			}
			else if (Character.isDigit(i)) {
				numberCount++;
			}
			else {
				otherCount++;
			}
		}
		System.out.println("Letter is : "+letterCount);
		System.out.println("Space is : "+spaceCount);
		System.out.println("Number is : "+numberCount);
		System.out.println("Other is : "+otherCount);
	}
}
```

## 练习题8
```java
import java.util.*;

public class Prac8 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		/*题目：一球从100米高度自由落下，
		 * 每次落地后反跳回原高度的一半；
		 * 再落下，求它在 第10次落地时，
		 * 共经过多少米？第10次反弹多高？ 
		 */
		ArrayList<Integer> leng = ballJump(100, 2);
		double total = 0.0;
		for(double i:leng) {
			total += i;
		}
		System.out.println("总共经过多少："
				+ ""+total);
		System.out.println("反弹多高："+leng.get(leng.size()-1)/4);
	}
	
	public static ArrayList<Integer> ballJump(int height, int n){
		ArrayList<Integer> arr = new ArrayList<Integer>();
		for (int i=0; i<n; i++) {
			if (i==0) 
				arr.add(height);
			else {
				height /= 2.0;
				arr.add(2*height);
			}
		}
		return arr;
	}

}
```

## 练习题9
```java
import java.util.*;

public class Prac9 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		/* 题目：有1、2、3、4个数字，
		 * 能组成多少个互不相同且无重复数字的三位数？都是多少？
		 */
		rankCombination();
	}
	public static void rankCombination(){
		ArrayList<Integer> arr = new ArrayList<Integer>();
		for (int i=1;i<=4;i++) {
			for (int j=i+1;j<=4;j++) {
				for (int k=j+1;k<=4;k++) {
					System.out.println(String.format("%d %d %d", i, j, k));
				}
			}
		}
	}
}
```

## 练习题10
```java

public class Prac10 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		/* 一个整数，它加上100和加上268后都
		 * 是一个完全平方数，请问该数是多少？
		 */
		for (double i=0.0;i<10000;i++) {
			double x = i+100;
			double y = i+268;
			if (isSquare(x) && isSquare(y))
				System.out.println((int) i);
		}
	}
	public static boolean isSquare(double x) {
		for (double i=0.0;i<(x/2);i++) {
			if(i*i==x)
				return true;
		}
		return false;
	}
	
}
```

## 练习题11
```java
import java.util.*;

public class Prac11 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		/* 题目：输入某年某月某日，判断这一天是这一年的第几天？
		 */
		whichDay();
		
	}
	public static void whichDay() {
		Scanner in = new Scanner(System.in);
		String intmp = in.nextLine().trim();
		String[] date = intmp.split("-");
		int year = Integer.parseInt(date[0]); //string 转化 int
		int month = Integer.parseInt(date[1]); //string 转化 int
		int day = Integer.parseInt(date[2]); //string 转化 int
		int two = 28;
		if(year%4==0 && year%100!=0) {
			two = 29;
		}
		int[] ls = {31,two,31,30,31,30,31,31,30,31,30,31};
		int daySum = 0;
		for(int i=0;i<month-1;i++) {
			daySum += ls[i];
		}
		daySum += day;
		System.out.printf("是第%d天", daySum);
	}	
}
```

## 练习题12
```java
import java.util.*;

public class Prac12 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		/* 输入三个整数x,y,z，请把这三个数由小到大输出。 
		 */
		printSort();
	}
	public static void printSort() {
		ArrayList<Integer> arr = new ArrayList<Integer>();
		System.out.println("请输入三个数,以逗号隔开:");
		Scanner in = new Scanner(System.in);
		String[] inStr = in.nextLine().trim().split(",");
		for (String i : inStr) {
			arr.add(Integer.parseInt(i));
		}
		Collections.sort(arr); //对Arraylist进行排序
		System.out.println("排序结果:");
		for(int i:arr) {
			System.out.println(i);
		}
	}

}
```
## 练习题13
```java

public class Prac13 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		/*
		 * 输出9*9口诀
		 */
		multipleTable();
	}
	public static void multipleTable() {
		int[] ls = {1,2,3,4,5,6,7,8,9};
		for (int i : ls) {
			for (int j : ls) {
				System.out.printf("%d*%d=%d\t", i,j,i*j);
			}
			System.out.printf("\n");
		}
	}

}
```

## 练习题14
```java

public class Prac14 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		// 题目：求1+2!+3!+...+20!的和 
		int sum=0;
		for (int i=1; i<21; i++) {
			sum += factorial(i);
		}
		System.out.println(sum);
	}
	public static int  factorial(int n) {
		if (n==1)
			return 1;
		return n*factorial(n-1); //递归求解
	}

}
```

# 剑指offer
## 1.查找重复数字
```java
/* 
在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。
请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。
*/

/* 
解题思路:由于题目的特殊性，即：长度为n的数组所有数字都在0~n-1范围内，使得可以将所有数字分配到对应为数字值所在的位置(numbers[i])。
因此使用类似hash查找的方式，不断替换位置与值不相等的数字，若此过程中发现需要替换的值在期待位置已经存在相同的值那么即为重复数字并保留，否则返回false。
还有一种情况即数组长度为0，那么无重复返回false。
*/

public class Solution {
    // Parameters:
    //    numbers:     an array of integers
    //    length:      the length of array numbers
    //    duplication: (Output) the duplicated number in the array number,length of duplication array is 1,so using duplication[0] = ? in implementation;
    //                  Here duplication like pointor in C/C++, duplication[0] equal *duplication in C/C++
    //    这里要特别注意~返回任意重复的一个，赋值duplication[0]
    // Return value:       true if the input is valid, and there are some duplications in the array number
    //                     otherwise false
    public boolean duplicate(int numbers[],int length,int [] duplication) {
        if (length==0){
            return false;
        }
        for (int i=0; i<length;i++){
            while (i!=numbers[i]){
                if (numbers[i]==numbers[numbers[i]]){
                    duplication[0] = numbers[i];
                    return true;
                }else{
                    swap(numbers, i, numbers[i]);
                }
            }
        }
        return false;
    }
    public void swap(int[] numbers, int i, int j){
        int tmp = numbers[i];
        numbers[i] = numbers[j];
        numbers[j] = tmp;
    }
}
```

## 2.二维数组中的查找
```java
/*
在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

二维数组: [[1,2,8,9],[2,4,9,12],[4,7,10,13],[6,8,11,15]]
整数: 7
返回: true
*/

/*
解题思路:按照题目的说明，在某个位置左/上方数字都比这个位置的值小，右/上方数字都比这个位置的值大。因此按照这个大小关系，
如果要搜寻某个值是否在二维数组中，只要拿这个数和当前位置的数字进行比较，如果target小则在左上找，如果target大则在右上
方找。但是这种寻找方式会产生两种选择(以小为例：上、左)，所以，初始位置应该设置在最右上角，这样寻找起来就会只有一种选择
（如果target小则往左也就是column-1，如果target大则往下也就是row+1）。最终如果找到返回ture，如果column<0或者
row==array.length则返回false
*/

public class Solution {
    public boolean Find(int target, int [][] array) {
        if (array==null || array.length==0 || (array.length==1 && array[0].length==0)){
            return false;
        }else{
            int col = array[0].length-1;
            int row = 0;
            boolean run = true;
            while (run){
                if (target < array[row][col]){
                    col -= 1;
                }else if(target > array[row][col]){
                    row += 1;
                }else{
                    return true;
                }
                 
                if (col==-1 || row==array.length){
                    run = false;
                }
            }
        }
        return false;
    }
}
```

## 3.从尾到头打印链表
```java
/*
输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。
*/

/*
解决思路:递归即可
*/

/**
*    public class ListNode {
*        int val;
*        ListNode next = null;
*
*        ListNode(int val) {
*            this.val = val;
*        }
*    }
*
*/

import java.util.ArrayList;
public class Solution {
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ArrayList<Integer> back = new ArrayList<Integer>();
        if (listNode!=null){
            back.addAll(printListFromTailToHead(listNode.next));//不是一个元素所以要addAll
            back.add(listNode.val);
        }
        return back;
    }
}
```

## 4.斐波那契数列
```java
/*
题目描述：
大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。
矩形覆盖也是相同的原理，因为矩形要么横着放，要么竖着放，f(n)=f(n-2)(横着放)+f(n-1)(竖着放)，一样的递推公式
*/

/*
解题思路：
斐波那契数列只依赖前两个值，因此，不断存储1-n的斐波那契数列，最后输出最后一个，时间复杂度低
*/

/*
问题：我们可以用 2*1 的小矩形横着或者竖着去覆盖更大的矩形。请问用 n 个 2*1 的小矩形无重叠地覆盖一个 2*n 的大矩形，总共有多少种方法？
思路：矩形覆盖也是相同的原理，因为矩形要么横着放，要么竖着放，f(n)=f(n-2)(横着放)+f(n-1)(竖着放)，一样的递推公式
*/

/*
问题：一只青蛙一次可以跳上 1 级台阶，也可以跳上 2 级。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。
思路：一样的递推公式
*/

import java.util.ArrayList;
public class Solution {
    public int Fibonacci(int n) {
        ArrayList<Integer> arr = new ArrayList<Integer>();
        arr.add(1);
        arr.add(1);
        int tmp;
        arr.size();
        if (n==0)
            return 0;
        if (n==1 || n==2)
            return 1;
        if (n>2){
            while (arr.size()<n){
                tmp = arr.get(arr.size()-2)+arr.get(arr.size()-1);
                arr.add(tmp);
            }
        }
        return arr.get(arr.size()-1);
    }
}
```

<hr />
