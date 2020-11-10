import setuptools

setuptools.setup(
    name='poisoning',
    version='0.1',
    python_requires='>=3.7',
    packages=setuptools.find_packages(),
    install_requires=['pytest>=5.4',
                      'numpy>=1.18',
                      'sklearn'                     
                      ],
    classifiers=[    
        'Programming Language :: Python :: 3.8',
    ],
)