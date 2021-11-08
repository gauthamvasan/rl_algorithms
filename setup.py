from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


setup(name='rl_algorithms',
      packages=find_packages(),
      install_requires=[],
      description='Simple, easy to use implementations of Reinforcement Learning (RL) algorithms',
      author='Gautham Vasan',
      url=' https://github.com/gauthamvasan/rl_algorithms',
      author_email='vasan@ualberta.ca',
      version='0.0.1')
