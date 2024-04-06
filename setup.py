from setuptools import find_packages, setup

setup(
    name='fin-stats-tools',
    version='0.0.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'fin-stats-tools=fin-stats-tools:main'
        ]
    }
)