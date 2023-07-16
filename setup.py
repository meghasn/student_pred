#responsible for creating ml model as a package

from setuptools import find_packages,setup

E_DOT='-e .'
def get_requirements(file_path):
    '''
    This function will return a list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","")for req in requirements]

        if E_DOT in requirements:
            requirements.remove(E_DOT)
    return requirements
setup(
name='mlproject',
version='0.0.1',
author='Megha',
author_email='mnair5@asu.edu',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')



)