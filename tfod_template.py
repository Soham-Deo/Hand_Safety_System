import os

directories = [
    "notebooks",
    "models",
    "src",
    "protos",
    "frozen_graph"
]

for dir_ in directories:
    os.makedirs(dir_, exist_ok=True)
    with open(os.path.join(dir_, ".gitkeep"), "w") as f:
        print("Directory {} Created Successfully".format(dir_))

files = [
    "params.yaml",
    os.path.join("src", "__init__.py"),
    "requirements.txt",
    "setup.py"
]

for file in files:
    with open(file, "w") as f:
        print("File {} Created Successfully".format(file))
