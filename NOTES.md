# Building and publishing

## Create a wheel 

Create a wheel with the setuptools -> Make sure you have wheel installed and run the code below in the main folder 

```bash
pip install wheel
python setup.py bdist_wheel
```

Generates the build and dist folders

## Create the source code binary file 

```bash
python setup.py sdist
```

Generates a source distribution file

