#!/usr/bin/env python
from setuptools import setup, find_packages


setup(
    name='hasheroku',
    version='0.0.1',
    author='Ivan Skorokhodov',
    author_email='iskorokhodov@gmail.com',
    url='https://github.com/universome/firelab',
    description='Experimental framework to run pytorch experiments',
    packages=find_packages(exclude=('tests',)),
    entry_points = {
        'console_scripts': ['firelab=firelab.cli:main'],
    },
    license='BSD',
    python_requires='>=3',
    long_description=open('README.md').read(),
    zip_safe=True
)
