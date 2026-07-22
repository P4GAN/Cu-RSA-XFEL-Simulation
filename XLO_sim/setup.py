from setuptools import setup, find_packages
from os import path, environ

cur_dir = path.abspath(path.dirname(__file__))

with open(path.join(cur_dir, 'requirements.txt'), 'r') as f:
    requirements = f.read().split()



setup(
    name='XLO_sim',
    version = 'v0.0.1',
    packages=find_packages(),  
    package_dir={'XLO_sim':'XLO_sim'},
    url='https://github.com/balticfish/XLO_sim',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=requirements,
    include_package_data=True,
    python_requires='>=3.6'
)
