#!/usr/bin/env python
# coding=utf-8

"""
python distribute file
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals, with_statement)

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_packages = f.readlines()


setup(
    name="BRAN",
    version="0.1.0",
    packages=find_packages(),
    dependency_links=[
        # If your project has dependencies on some internal packages that is
        # not on PyPI, you may list package index url here. Then you can just
        # mention package name and version in requirements.txt file.
    ],
    install_requires=install_packages,
    package_data={
        'dataset': ['*']
    },
    # data_files=[('config', ['bran/config/Dockerfile'])],
    author="M S Shankar",
    author_email="m.s.shankar13689@gmail.com",
    maintainer1="Daniel Morales",
    maintainer1_email="d.morales@kigroup.de",
    maintainer2="Anton Ivanov",
    maintainer2_email="a.ivanov@kigroup.de",
    description="BRAN (Basic Recognition and Authorisation at eNtrance)",
    long_description=open('README.md').read(),
    entry_points={
        'console_scripts': ['bran = bran.__main__:start']
    },
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',
        'Programming Language :: Python :: 3.7',
    ]
)
