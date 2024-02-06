#
# population_mcmc setuptools script
#
from setuptools import setup, find_packages


def get_version():
    """
    Get version number from the population_mcmc module.
    """
    import os
    import sys

    sys.path.append(os.path.abspath('population_mcmc'))
    version = "1.0.0"
    sys.path.pop()

    return version


def get_requirements():
    requirements = []
    with open("requirements.txt", "r") as file:
        for line in file:
            requirements.append(line)
    return requirements


setup(
    # Module name
    name='population_mcmc',

    # Version
    version=get_version(),

    description='An implementation of a Population MCMC algorithm',

    maintainer='Matthew Ghosh',

    maintainer_email='matthew.ghosh@gtc.ox.ac.uk',

    url='https://github.com/mghosh00/PopulationMCMC',

    # Packages to include
    packages=find_packages(include=('population_mcmc', 'population_mcmc.*')),

    # List of dependencies
    install_requires=get_requirements(),

    extras_require={
        'docs': [
            'sphinx>=1.5, !=1.7.3',
            'sphinx_rtd_theme',
        ],
        'dev': [
            'flake8>=3',
            'pytest',
            'pytest-cov',
        ],
    },
)
