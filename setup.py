from setuptools import setup, find_packages

setup(name='xwhy',
      version='0.0.0.1',
      description='Explain Why (XWhy) with Statistical Model-agnostic Interpretability with Local Explanations (SMILE)',
      url='https://github.com/Dependable-Intelligent-Systems-Lab/xwhy',
      author='Mojgan Hashemian, Koorosh Aslansefat (corresponding), Mohammad Naveed Akram, Ioannis Sorokos, Martin Walker, Yiannis Papadopoulos',
      author_email='koo.ec2008@gmail.com',
      license='BSD',
      packages=find_packages(exclude=['js', 'node_modules', 'tests']),
      python_requires='>=3.5',
      install_requires=[
          'matplotlib',
          'numpy',
          'scipy',
          'tqdm >= 4.29.1',
          'scikit-learn>=0.18',
          'scikit-image>=0.12',
          'pyDOE2==1.3.0',
          'twine==1.13.0'
      ],
      extras_require={
          'dev': ['pytest', 'flake8'],
      },
      include_package_data=True,
      zip_safe=False)
