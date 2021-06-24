#python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


path ="/mnt/c/Users/b_tib/coding/Msc/oLINGI2262/ml-classification-evaluation/decision-trees-bagging-randomforest/datasets"

train_df = pd.read_csv(str(path) + "/BostonHouseTrain.csv", index_col=0)
test_df = pd.read_csv(str(path) + "/BostonHouseTest.csv", index_col=0)

frame = pd.DataFrame(columns = ["min_samples_split", "NodeCount", "TrainAcc", "TestAcc"])

#%%
############################
# Pre-prunning
############################

fracs = [1] # [0.05, 0.1, 0.2, 0.5, 0.99]
prunes=[0.01, 0.05, 0.5]

i=0
run=0
for f in fracs:
    for prune in prunes:
        for run in np.arange(1):

            train_df_frac = train_df.sample(frac=f,random_state=i)

            X_train = train_df_frac.iloc[:,:-1]
            Y_train = train_df_frac.iloc[:,-1:]

            X_test = test_df.iloc[:,:-1]
            Y_test = test_df.iloc[:,-1:]

            clf = DecisionTreeClassifier(min_samples_split=prune,random_state=0)
            clf = clf.fit(X_train, Y_train)

            node = clf.tree_
            score_train = clf.score(X_train,Y_train)
            score_test = clf.score(X_test,Y_test)

            frame.loc[i] = [prune, node.node_count, score_train, score_test]
            i+=1

print(frame)


#%%
############################
# Post-prunning
############################

def plot_impurity_leafs(ccp_alphas, impurities):
	fig, ax = plt.subplots()
	ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
	ax.set_xlabel("effective alpha")
	ax.set_ylabel("total impurity of leaves")
	ax.set_title("Total Impurity vs effective alpha for training set")


def prunning_impurity(ccp_alphas, X_train, Y_train):
	clfs = [] # meta parameters configuration
	for ccp_alpha in ccp_alphas: # parameter values
		clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
		clf.fit(X_train, Y_train)
		clfs.append(clf)
	return clfs

def plot_tree_vs_alpha(clfs):
	node_counts = [clf.tree_.node_count for clf in clfs]
	depth = [clf.tree_.max_depth for clf in clfs]
	fig, ax = plt.subplots(2, 1)
	ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
	ax[0].set_xlabel("alpha")
	ax[0].set_ylabel("number of nodes")
	ax[0].set_title("Number of nodes vs alpha")
	ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
	ax[1].set_xlabel("alpha")
	ax[1].set_ylabel("depth of tree")
	ax[1].set_title("Depth vs alpha")
	fig.tight_layout()

def plot_accuracies(clfs):
	train_scores = [clf.score(X_train, Y_train) for clf in clfs]
	test_scores = [clf.score(X_test, Y_test) for clf in clfs]

	fig, ax = plt.subplots()
	ax.set_xlabel("alpha")
	ax.set_ylabel("accuracy")
	ax.set_title("Accuracy vs alpha for training and testing sets")
	ax.plot(ccp_alphas, train_scores, marker='o', label="train",
			drawstyle="steps-post")
	ax.plot(ccp_alphas, test_scores, marker='o', label="test",
			drawstyle="steps-post")
	ax.legend()
	plt.show()

	print('Balanced training and test accuracy /n')
	frame = pd.DataFrame(columns = ["ccp_alphas","NodeCount", "TrainAcc", "TestAcc"])
	row = 0
	for i in range(len(test_scores)-1):
		if abs(train_scores[i] - test_scores[i]) < 0.02:
			frame.loc[row] = [ccp_alphas[i], clfs[i].tree_.node_count, train_scores[i], test_scores[i]]
			row += 1
	print(frame)

path ="/mnt/c/Users/b_tib/coding/Msc/oLINGI2262/ml-classification-evaluation/decision-trees-bagging-randomforest/datasets"

train_df = pd.read_csv(str(path) + "/BostonHouseTrain.csv", index_col=0)
test_df = pd.read_csv(str(path) + "/BostonHouseTest.csv", index_col=0)

X_train = train_df.iloc[:,:-1]
Y_train = train_df.iloc[:,-1:]

clf = DecisionTreeClassifier(random_state=0)

path = clf.cost_complexity_pruning_path(X_train, Y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

plot_impurity_leafs(ccp_alphas, impurities) # total impurity by alpha
clfs = prunning_impurity(ccp_alphas, X_train, Y_train) # generates same tree strcuture with dif. ccp_alpha
plot_tree_vs_alpha(clfs) # how tree nodes and depth varies with alpha
plot_accuracies(clfs) # plot train and test accuracies according to alpha



# %%
