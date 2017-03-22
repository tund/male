for i in range(6):
    print("Test %d" % (i+1))
    exec(open("test_case%d.py" % (i+1), 'rt').read())
    print("============================")