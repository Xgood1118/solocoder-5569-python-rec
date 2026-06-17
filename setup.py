from setuptools import setup, find_packages

setup(
    name='rec_toolkit',
    version='0.1.0',
    description='Lightweight recommendation system toolkit for teaching and small business',
    author='rec_toolkit team',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'flask>=2.3.0',
        'flask-cors>=4.0.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'annoy>=1.17.0',
        'jieba>=0.42.1',
        'apscheduler>=3.10.0',
        'click>=8.1.0',
        'pyyaml>=6.0',
        'mlxtend>=0.23.0',
    ],
    entry_points={
        'console_scripts': [
            'rec-toolkit=rec_toolkit.cli:main',
        ],
    },
    python_requires='>=3.8',
)
