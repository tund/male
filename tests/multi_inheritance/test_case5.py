__metaclass__ = type
class A:
    def __init__(self):
        print("A")

class B(A):
    def __init__(self):
        super(B, self).__init__()
        print("B")

class C(A):
    def __init__(self):
        super(C, self).__init__()
        print("C")

class D(B):
    def __init__(self):
        super(D, self).__init__()
        print("D")


class X(D, C):
    def __init__(self):
        super(X, self).__init__()
        print("X")
print("Inheritance structure:")
print("[ * means calling super.]")
print("          A              ")
print("        /   \            ")
print("       B*    C*          ")
print("       |    /            ")
print("       D*  /             ")
print("       |  /              ")
print("       X*                ")
obj = X()
print("Method Resolution Order:")
print(X.mro())
print("Explanation:")
print("1. Depth first search: X -> D -> B -> A -> C -> A")
print("2. Remove superclasses in the middle: X -> D -> B -> C -> A")
print('Reference: http://stackoverflow.com/questions/3634211/python-3-1-c3-method-resolution-order')
