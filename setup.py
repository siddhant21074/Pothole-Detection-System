## Used to build out application as a packages and deploy


from setuptools import find_packages,setup

def get_requirenments(file_path:str)->list[str]:
    '''
        Installs the dependencies
    '''
    HYHEN_E_DOT = '-e .'
    with open(file_path) as file:
        require = file.readline()
        require = [req.replace("\n"," ") for req in require ]

        if HYHEN_E_DOT in require:
            require.remove(HYHEN_E_DOT)
    return require
setup(
    name="mlproject",
    author="siddhant",
    author_email="siddhantthombare2121@gamil.com",
    version='0.0.1',
    packages=find_packages(),
    install_requires= get_requirenments('requirenments.txt')
)     