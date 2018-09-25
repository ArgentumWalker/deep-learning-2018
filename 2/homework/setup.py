from distutils.core import setup

setup(
    name='ResnextHW',
    version='0.1.0',
    author='ArgentumWalker',
    packages=['resnext', 'resnext.test'],
    description='PyTorch resnext implementation',
    install_requires=[
        "pytorch >= 0.4.1",
        "torchvision >= 0.2.1",
        "tensorboard >= 1.8.0",
        "tensorboardX >= 1.2",
        'sklearn',
        'numpy'
    ]
)
