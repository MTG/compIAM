try: # for pip >= 10
    from pip._internal.req import parse_requirements
    import uuid
    install_reqs = parse_requirements('requirements.txt', session=uuid.uuid1())
    reqs = [str(req.req) for req in install_reqs]
except: # for pip <= 9.0.3
    def parse_requirements(filename):
        """ load requirements from a pip requirements file"""
        lineiter = (line.strip() for line in open(filename))
        return [line for line in lineiter if line and not line.startswith("#")]
    reqs = parse_requirements('requirements.txt')

from setuptools import find_packages, setup

setup(
    name='compIAM',
    version="1.0",
    packages=find_packages(),
    author_email=['thomas.nuttall@upf.edu','genis.plaja@upf.edu'],
    zip_safe=False,
    include_package_data=True,
    long_description=open('README.md').read(),
    install_requires=reqs
)