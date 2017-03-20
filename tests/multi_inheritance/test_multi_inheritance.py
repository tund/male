for i in range(6):
    print("Test %d" % (i+1))
    execfile("test_case%d.py" % (i+1))
    print("============================")