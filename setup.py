from setuptools import find_packages, setup

if __name__ == '__main__':
    setup(
        name = 'lamtk',
        version = '0.1.0',
        description = 'LiDAR aggregation and mapping toolkit',
        author = 'Chengjie Huang, Benjamin Therien',
        author_email = 'c.huang@uwaterloo.ca',
        url = 'https://github.com/c7huang/lamtk',
        packages = find_packages(include=['lamtk'])
    )
