#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = ['pytest>=3', ]

setup(
    author="Eduardo Maldonado-Cruz",
    author_email='emaldonadocruz@utexas.edu',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Education',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Uncertainty Tuning (UTuning) is a package that focuses on summarizing uncertainty model performance for optimum hyperparameter tuning by using the uncertainty model goodness metric.",
    entry_points={
        'console_scripts': [
            'UTuning=UTuning.cli:main',
        ],
    },
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='UTuning',
    name='UTuning',
    packages=find_packages(include=['UTuning', 'UTuning.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/emaldonadocruz/UTuning',
    version='0.0.8',
    zip_safe=False,
)
