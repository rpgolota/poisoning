import setuptools

def get_version():
    exec(open('poisoning/version.py').read(), locals())
    return locals()['__version__']

def get_requirements():
    with open('requirements.txt') as f:
        lines = f.read()
        
    lines = lines.strip().split('\n')
    return lines

name = 'poisoning'

version = get_version()

description = 'Poisoning datasets using gradient ascent, targeting feature selection.'

url='https://github.com/rpgolota/poisoning/'

python_requires = '>=3.7'

packages = ['poisoning']

install_requires = get_requirements()

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