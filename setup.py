from setuptools import find_packages, setup

setup(
    name = 'Medical Chatbot',
    version= '0.0.0',
    author= 'Saarthak Gupta',
    author_email= 'saarthak.gupta.mec22@itbhu.ac.in',
    packages= find_packages(),
    install_requires = []

)

## setup.py file get executed automatically while install requirement.txt file because we have added -e . as a requirement