import setuptools

def get_version():
    exec(open('poisoning/version.py').read(), locals())
    return locals()['__version__']

name = 'poisoning'

version = get_version()

description = 'Poisoning datasets using gradient ascent, targeting feature selection.'

url='https://github.com/rpgolota/poisoning/'

python_requires = '>=3.7'

packages = ['poisoning']

install_requires = ['numpy>=1.18,<=1.19.3',
                    'sklearn']

classifiers = [ 'Development Status :: 5 - Production/Stable',
                'Intended Audience :: Science/Research',
                'Operating System :: OS Independent',
                'Programming Language :: Python :: 3.8']

setuptools.setup(
    name=name,
    version=version,
    description=description,
    url=url,
    python_requires=python_requires,
    packages=packages,
    install_requires=install_requires,
    classifiers=classifiers,
)