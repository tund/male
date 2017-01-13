from distutils.core import setup
from setuptools import find_packages

install_requires = [
    'numpy',
    'matplotlib',
    'scikit-learn',
    'scipy'
]

setup(
    name='male',
    version='0.1.0',
    url='https://github.com/tund/male',
    license='MIT',
    author='tund',
    author_email='nguyendinhtu@gmail.com',
    description='MAchine LEarning (MALE)',
    packages=find_packages(),
    install_requires=install_requires,
)
