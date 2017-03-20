class A:
    def __init__(self):
        print("A")

class B:
    def __init__(self):
        print("B")

class C(A, B):
    def __init__(self):
        super(C, self).__init__()
        print("C")

class D(A, B):
    def __init__(self):
        super(D, self).__init__()
        print("D")


class X(C, D):
    def __init__(self):
        super(X, self).__init__()
        print("X")
print("Inheritance structure:")
print("[ * means calling super.]")
print("       A    B            ")
print("       | \/ |            ")
print("       | /\ |            ")
print("       C*   D*           ")
print("        \  /             ")
print("         X*              ")
print(" Method Resolution Orther:")
print(X.mro())
print("Output:")
obj = X()
print("Explanation:")
print("1. Depth first search: X -> C -> A -> B -> D -> A -> B")
print("2. Remove superclasses in the middle: X -> C -> D -> A -> B")
print('Reference: http://stackoverflow.com/questions/3634211/python-3-1-c3-method-resolution-order')