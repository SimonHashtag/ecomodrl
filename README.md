# EcoModRL
Economic modeling meets deep reinforcement learning - an empirical analysis

### 1. Project setup

#### 1.1 conda-lock
We use conda-lock to install the requirements of this project. 
This ensures full reproducibility of the code.

First install conda-lock via condax (safer):
```
conda install condax
condax install conda-lock
```

Create a new conda environment "YOURENV" or install the packages into your preexisting conda env "YOURENV" by
```
conda-lock install -n YOURENV requirements.conda-lock.yml
```

To update a single package to the latest version compatible with the version constraints in the source:
```
conda-lock lock  --lockfile requirements.conda-lock.yml --update PACKAGE
```

To re-solve the entire environment (produce a new requirements.conda-lock.yml), e.g. after changing a version constraint or adding a package in the source file:
```
conda-lock -f requirements.yml --lockfile requirements.conda-lock.yml
```

#### 1.2 pip-compile
You can also use pip to reproduce our results.
Simply run:
```
pip install -r requirements.txt
```

If you want to make your work reproducible as well, follow these steps:
1. First install pip-tools
2. If you would like to add a new module to the project add it to requirements.in.
3. Then you can update requirements.txt with the given requirements.in file.

The code:
```
pip install pip-tools
pip-compile requirements.in
pip install --upgrade pip
pip install -r requirements.txt
```

To update all modules in your requirements.txt file:
```
pip-compile --upgrade
```

#### 1.2 pre-commit
We use pre-commit for code style and preventing large files in our project.
Install pre-commit in git hooks to check if our code and files
are in line with our definitions in pre-commit-config.yaml.
```
pre-commit install
```
Now at each commit pre-commit checks our files

#### 1.3 TensorBoard
To enable hosting, live tracking and sharing of your experiments you can use TensorBoard. <br>
###### 1.3.1 TensorBoard Dev (Image display not supported!)
To upload your experiment on *Tensorboard Dev* enter your terminal and type:

```
tensorboard dev upload --logdir=\relative\path\to\logs (default: 'runs\..') \
      --name "(optional) Cool experiment" \
      --description "(optional) The agent learns her optimal intertemporal consumption."
```

After accepting to the condition that your data will be visible to everyone (who has the link), you can
enter the supplied link to the search bar of your browser and view your experiment online.

###### 1.3.2 Local TensorBoard

In order to enable display of images and figures you can use Tensorboard locally. To do so type into the terminal:
```
tensorboard --logdir=\relative\path\to\logs (default: 'runs\..') --window_title 'Name of tab in browser'
```
After hitting the supplied link you can see the training progress in your browser. <br>
Sharing of the results is only possible with TensorBoard Dev.

TensorBoard automatically downsamples logged summaries to prevent out-of-memory errors. If you wish to obtain
control over downsampling, add the following flag to the terminal command with the category of the uploaded summary
and the number of frames you wish to keep: <br>
e.g. ```--samples_per_plugin "scalars=10000, images=100"```
