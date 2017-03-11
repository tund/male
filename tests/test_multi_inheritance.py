class A:
    def __init__(self):
        print("A")

class B(A):
    def __init__(self):
        super(B, self).__init__()
        print("B")

class C(A):
    def __init__(self):
        print("C")

class D(B, C):
    def __init__(self):
        super(D, self).__init__()
        print("D")
print(D.mro())
obj = D()
