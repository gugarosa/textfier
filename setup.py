from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='textfier',
      version='1.0.0',
      description='Text-based Modifiers',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Gustavo Rosa',
      author_email='gustavo.rosa@unesp.br',
      url='https://github.com/gugarosa/textfier',
      license='GPL-3.0',
      install_requires=['nltk>=3.5',
                        'pylint>=2.5.2',
                        'scikit-learn>=0.22.2',
                        'transformers>=3.0.2'
                        ],
      extras_require={
          'tests': ['coverage',
                    'pytest',
                    'pytest-pep8',
                    ],
      },
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
