#!/usr/bin/env python

from setuptools import setup

REQUIRED_PACKAGES = [
      "pyyaml"
]

setup(name='COSSMO',
      version='1.0',
      author='Hannes Bretschneider',
      author_email='hannes@psi.utoronto.ca',
      url='https://github.com/PSI-Lab/COSSMO',
      packages=['cossmo'],
      scripts=['bin/train_cossmo.py'],
      include_package_data=True,
      install_requires=REQUIRED_PACKAGES
)
