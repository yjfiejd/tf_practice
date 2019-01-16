# @TIME : 2019/1/13 下午2:36
# @File : super_learn.py

# super 的一个最常见用法可以说是在子类中调用父类的初始化方法了，比如：
# 理解super的本质，事实上，super 和父类没有实质性的关联。


class Base(object):
    def __init__(self):
        print("enter Base")
        print("leave Base")


class A(Base):
    def __init__(self):
        print("enter A")
        super(A, self).__init__()
        print("leave A")


class B(Base):
    def __init__(self):
        print("enter B")
        super(B, self).__init__()
        print("leave B")


class C(A, B):
    def __init__(self):
        print("enter C")
        super(C, self).__init__()
        print("leave C")

# print(C.mro())
# # http://funhacks.net/explore-python/Class/super.html
#
# c = C()
# print('c', c)


class Base1(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        print('enter base1')
class A(Base1):
    def __init__(self, a, b, c):
        super(A, self).__init__(a, b)
        self.c = c
        print('enter A()')


a = A(1, 2, 3)
print(a)