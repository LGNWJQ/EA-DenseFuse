# EA-DenseFuse
> 改进DenseFuse：加入注意力机制和边缘损失函数

## 说明

- 本代码库的实现为**电子科技大学-信息与通信工程学院-通信工程系-综合课程设计**的一部分
- 这是对DenseFuse的进一步改进：
  - 加入边缘损失函数
  - 加入空间注意力和通道注意力机制



## 使用方法

可见[DenseFuse-Refactoring-of-PyTorch](https://github.com/LGNWJQ/DenseFuse-Refactoring-of-PyTorch)



##  改进方式

### 加入边缘损失函数

![](https://raw.githubusercontent.com/LGNWJQ/picgo/main/EA-DenseFuse/f1.jpg)

* 效果

![](https://raw.githubusercontent.com/LGNWJQ/picgo/main/EA-DenseFuse/f2.jpg)

![](https://raw.githubusercontent.com/LGNWJQ/picgo/main/EA-DenseFuse/f3.jpg)



### 加入注意力机制

* 使用经典的**空间注意力+通道注意力**机制：[CBAM](https://arxiv.org/pdf/1807.06521.pdf)

![](https://raw.githubusercontent.com/LGNWJQ/picgo/main/EA-DenseFuse/CBAM.jpg)

* 改进效果

![](https://raw.githubusercontent.com/LGNWJQ/picgo/main/EA-DenseFuse/f4.jpg)

![](https://raw.githubusercontent.com/LGNWJQ/picgo/main/EA-DenseFuse/f5.jpg)



















