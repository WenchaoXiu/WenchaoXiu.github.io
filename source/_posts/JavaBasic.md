---
title: Java基本操作
tags: []
date: 2018-08-03 17:01:18
permalink:
categories: Java
description: 对Java基本的操作进行总结
image:
---
<p class="description"></p>

<!-- more -->

# 字符串
## 字符串比较：
```java
	String str = "qdaw"
	str.compareTo(anotherString)
	str.compareToIgnoreCase(anotherString)
```

## 字符串最后一个匹配索引
```java
	String str = "dada"
	str.lastIndexOf("a") //返回-1或者对应的index
	str.indexOf("a") //返回-1或者对应的index
```

## 字符串的截取
```java
	s.substring(0, pos) //左闭右开
	s.substring(3) //index 3到最后
```

## 字符串的反转
```java
	String str = "abc"
	String rev = new StringBuffer(str).reverse().toString()
```

## 字符串分割
```java
	String[] temp：
	String str = "add.adw.wadw"
	String delimer = "\\."
	temp = str.split(delimer) //利用.对字符串分割

	String str = "daw,wdad,wdadwa";
	StringTokenizer st2 = new StringTokenizer(str, ",");
	while (st2.hasMoreElements()){
		System.out.println(st2.nextElement());
	}
```

## 字符串大小写转换
```java
	String str = "abcd";
	String strUp = str.toUpperCase();
	String strLow = str.toLowerCase();
```

## 字符串格式
```java
	%n 换行符，%s 字符串，%c 字符
	%d 十进制整数，%x 16进制整数，%o 8进制整数
	%f 浮点数，%b 布尔类型
```


# 数组操作
## 数组排序
```java
	int[] arr = {-1,-2,3,4,5,-5};
	Array.sort(arr); //没有返回值，arr直接就是排好序的
```

## 数组查找
```java
	int[] arr = {-1,-2,3,4,5,-5};
	int index = Array.binarySearch(arr, 3); //binarySearch需要先对数组排序,返回对应的位置或者负数
```

## 数组反转
```java
	ArrayList<String> arrayList = new ArrayList<String>();
	arrayList.add("A");
	arrayList.add("B");
	Collections.reverse(arrayList); // 对元素反转
```

## 数组最大最小值
```java
	Integer[] numbers = { 8, 2, 7, 1, 4, 9, 5};
    int min = (int) Collections.min(Arrays.asList(numbers));
    int max = (int) Collections.max(Arrays.asList(numbers));
```

## 数组合并
```java
	String a[] = { "A", "E", "I" };
    String b[] = { "O", "U" };
    List list = new ArrayList(Arrays.asList(a));
    list.addAll(Arrays.asList(b));
    Object[] c = list.toArray();
    System.out.println(Arrays.toString(c));
```

## 数组填充
```java
	int[] arr = new int[6];
	Array.fill(arr, 100); //数组进行填充，所有元素都为100
	Array.fill(arr, 3, 6, 50); //在index3，4，5上填充50
```

## 数组删除
```java
	ArrayList<String> objArray = new ArrayList<String>();
	objArray.add(0,"第 0 个元素");
    objArray.add(1,"第 1 个元素");
    objArray.add(2,"第 2 个元素");
    objArray.remove(1); //通过index进行删除
    objArray.remove("第 0 个元素") //通过内容删除
```

## 数组添加元素
```java
	ArrayList<String> list = new ArrayList<String>();
	list.add(2, "Item3") //在数组的第三个位置添加"Item3"
```

## 数组求差集,交集,以及是否含有某个元素
```java
	ArrayList objArray = new ArrayList();
    ArrayList objArray2 = new ArrayList();
    objArray2.add(0,"common1");
    objArray2.add(1,"common2");
    objArray2.add(2,"notcommon");
    objArray2.add(3,"notcommon1");
    objArray.add(0,"common1");
    objArray.add(1,"common2");
    objArray.add(2,"notcommon2");
    System.out.println("array1 的元素" +objArray);
    System.out.println("array2 的元素" +objArray2);
    objArray.removeAll(objArray2); // 数组差集，结果为objArray
    objArray.retainAll(objArray2); // 数组求交集
    objArray.contains("common2"); // 数组是否包含某个元素
```

## 数组求并集
```java
	String[] str1 = {"1", "2", "3", "4"};
	String[] str2 = {"2", "3", "4", "5"};
	Hashset<String> set = new Hashset<String>();
	for (String i : str1){
		set.add(i);
	}
	for (String i : str2){
		set.add(i);
	}
	String[] ret = {};
	set.toArray(ret); // ret即为利用Hashset获取的两个数组的并集
```

## 判断数组是否相等
```java
	int[] ary = {1,2,3,4,5,6};
    int[] ary1 = {1,2,3,4,5,6};
    int[] ary2 = {1,2,3,4};
    Arrays.equals(ary, ary1); //判断两个数组是否相等
```




<hr />
