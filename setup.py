
from setuptools import find_packages,setup
from typing import List

E_DOT="-e ."

def get_requirement(file_path:str)->List[str]:
    '''this function will return the list of requirements'''
    requiremmets=[]
    with open(file_path) as file_obj:
        requiremmets=file_obj.readlines()
        requiremmets=[req.replace("\n","") for req in requiremmets]
        if E_DOT in requiremmets:
            requiremmets.remove(E_DOT)
    return requiremmets

setup(
    name="Ml project",
    version="0.0.1",
    author="AbuZayDin",
    author_email="olamidehassan007@gmail.com",
    packages=find_packages(),
    install_requires=get_requirement('requirements.txt')
)

