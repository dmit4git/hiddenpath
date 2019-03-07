from setuptools import setup

setup(name='hiddenpath',
      version='0.1',
      description='Customizable OpenAI Gym Environment with asynchronous visualisation',
      url='http://github.com/dmit4git/hiddenpath',
      author='Dmitry Sherbakov',
      author_email='box4dmitry@gmail.com',
      license='MIT',
      packages=['hiddenpath'],
      install_requires=[
          'gym', 'pyglet',
      ],
      zip_safe=False)
