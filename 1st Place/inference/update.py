import os
os.system("pip install -qq timm-0.6.13-py3-none-any.whl")
os.system('python setup.py develop && python examples/hello-world.py >/dev/null 2>&1')
