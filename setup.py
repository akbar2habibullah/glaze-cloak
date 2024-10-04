from setuptools import setup, find_packages

setup(
    name='glaze-cloak',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'pillow',
        'einops',
        'diffusers',
        'requests',
        'tqdm',
        'numpy',
    ],
    author='Habibullah Akbar',
    author_email='Akbar2habibullah@gmail.com',
    description='Implementation of the Glaze image cloaking algorithm for protecting against unauthorized AI training',
    long_description=open("README.md").read(), 
    url='https://github.com/akbar2habibullah/glaze-cloak', 
    license='GPL-3.0 license',
)