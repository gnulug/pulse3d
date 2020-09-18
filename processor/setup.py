from setuptools import setup

setup(
    name='pulse3d-processor',
    version='0.1.0',
    description='Processor that projects input sound in 3D',
    url='https://github.com/gnulug/pulse3d',
    author='gnulug contributors',
    author_email='glug@acm.uiuc.edu',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='pulseaudio audio 3d processor',
    py_modules=['pulse3d_processor'],
    install_requires=[
        'numpy',
        'scipy',
        'pyaudio',
        'prettyparse'
    ],
    entry_points={
        'console_scripts': [
            'pulse3d-processor=pulse3d_processor:main'
        ],
    }
)
