import pytest


@pytest.mark.skip(reason="could not run this function using pytest")
def test_multiinherit():
    for i in range(6):
        print("Test %d" % (i + 1))
        exec(open("test_case%d.py" % (i + 1), 'rt').read())
        print("============================")


if __name__ == '__main__':
    pytest.main([__file__])
    # test_multiinherit()
