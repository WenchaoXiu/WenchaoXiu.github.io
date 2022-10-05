---
title: Hexo搭建博客
tags: []
date: 2018-07-12 16:07:52
permalink:
categories: Website
description: 这里主要记录搭建blog时使用的命令
image:
---
<p class="description">记录一下如何使用hexo进行博客撰写</p>

<!-- more -->

## 本地调试
```bash
hexo s --debug -p 5000
```
## 网上更新
```bash
hexo clean
hexo g
hexo d
```
## 新建md文件
```bash
hexo new post md_name
```
## 插入图片
```bash
# 将图片放入~/blog/public/images文件夹下，width可以用来调整图片长宽比例
<div style="width: 300px; margin: auto">![Git区域](/images/git_learning/gitzone.png)</div>
```
[图片插入参考文章](https://yanyinhong.github.io/2017/05/02/How-to-insert-image-in-hexo-post/)


更多[hexo命令](http://moxfive.xyz/2015/12/21/common-hexo-commands/)
更多内容参见[blog](https://reuixiy.github.io/technology/computer/computer-aided-art/2017/06/09/hexo-next-optimization.html)
对于markdown的使用可以参考这篇[blog](https://www.ofind.cn/blog/HEXO/HEXO%E4%B8%8B%E7%9A%84Markdown%E8%AF%AD%E6%B3%95%28GFM%29%E5%86%99%E5%8D%9A%E5%AE%A2.html)
<hr />