sudo apt-get install -qq libblas-dev liblapack-dev
pip install -U setuptools pip wheel
git clone https://github.com/astaric/orange3-requirements wheelhouse
pip install wheelhouse/nose-1.2.1-py3-none-any.whl wheelhouse/scipy-0.14.0-cp34-cp34m-linux_x86_64.whl

python setup.py build_ext -i

