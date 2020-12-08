from sklearn.datasets import load_digits, fetch_openml, fetch_olivetti_faces, fetch_20newsgroups

def _fetch_mnist():
	return load_digits()

def _fetch_fasion_mnist():
	return fetch_openml(name="Fashion-MNIST")

def _fetch_olivetti():
	return fetch_olivetti_faces()

def _fetch_coil():
	raise Exception("20newsgroups not supported at the moment!")
	# Ref: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html
	return fetch_20newsgroups()