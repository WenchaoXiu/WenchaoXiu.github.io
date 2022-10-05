---
title: Vim命令学习
tags: []
date: 2020-02-25 13:17:18
permalink:
categories: Vim
description: Vim基本命令使用
image:
---
<p class="description">这里记录常见的Vim命令，后续会随着使用不断增加</p>

<!-- <img src="https://" alt="" style="width:100%" />
 -->

<!-- more -->

### 基本Vim命令
1. 正常模式与编辑模式切换
```bash
vi xxx # 利用vim打开文件，此时处于正常模式下
esc键 # 退出编辑模式到正常模式
```

2. 删除
```bash
x # 删除光标之后的一个字符
dd # 删除一行
u # 撤销操作
```

3. 复制
```bash
yy # 复制光标所在行到缓冲区，yy前可加数字，例如：6yy表示复制6行
p # 将yy复制好的缓冲区内容粘贴到光标所在位置
```

4. 插入文字
```bash
i # 对文件进行插入编辑
```

5. 保存
```bash
:q! # 为文件的编辑不保存退出
:wq! # 对文件的编辑保存退出
```

6. 查找
```bash
/xxx # 查找xxx字符，使用n进行下一次查找的跳转
```

7. 替换
```bash
:%s/old/new/g # 对全文中的old替换为new
:.,+2s/old/new/g # 替换当前行与接下来两行
:5,12s/old/new/g # 替换5-12行里面的内容
```

8. 翻页
```bash
Ctrl+f # 向下翻一页
Ctrl+d # 向下翻半页
Ctrl+b # 向上翻一页
Ctrl+u # 向上翻半页
```
<br>


### Vim命令图谱(网上摘来的)
<div style="width: 750px; margin: auto">![Vim命令图谱](vim.jpg)</div>


<hr />
