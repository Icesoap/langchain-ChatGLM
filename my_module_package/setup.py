from setuptools import setup

#此方法不好用
#from distutils.core import setup
# setup(name='langchain', version='0.0.323', author='yuquan')
# setup(name='langchain', version='0.0.323', author='yuquan', packages=[''], package_dir={'': 'langchain'})

setup(
       name='langchain',
       version='0.0.323',
       description='langchain-0.0.323 custom code',
       author='yuquan',
       # author_email='your@email.com',
       # packages=['your_package'],
       install_requires=[
           # List your dependencies here
       ],
   )
