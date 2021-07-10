This repository implements the LOFT approach described in the IROS 2021 paper:

Learning Symbolic Operators for Task and Motion Planning\
Tom Silver*, Rohan Chitnis*, Joshua Tenenbaum, Leslie Pack Kaelbling, Tomas Lozano-Perez\
Link to paper: https://arxiv.org/abs/2103.00589

Instructions for running (tested on OS X and Ubuntu 18.04):
* Use Python 3.6 or higher.
* Download Python dependencies: `pip install -r requirements.txt`.
* Download the NDR package to a location on your path: https://github.com/tomsilver/ndr

Now, `./run.sh` should work, and should finish in less than a second.
You should see the printout `In total, solved 30 / 30` near the end.
The three different environments can be run by changing the `ENV`
variable in run.sh. Data has been included already in the `data/`
folder, but if you would like to regenerate it, you can set
`COLLECT_DATA=1` in run.sh. All three environments should yield 100%
test success rate on all seeds.

For questions or comments, please email tslvr@mit.edu and ronuchit@mit.edu.