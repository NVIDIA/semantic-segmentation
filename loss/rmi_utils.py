# This code is adapted from: https://github.com/ZJULearning/RMI

# python 2.X, 3.X compatibility
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn.functional as F


__all__ = ['map_get_pairs', 'log_det_by_cholesky']


def map_get_pairs(labels_4D, probs_4D, radius=3, is_combine=True):
	"""get map pairs
	Args:
		labels_4D	:	labels, shape [N, C, H, W]
		probs_4D	:	probabilities, shape [N, C, H, W]
		radius		:	the square radius
	Return:
		tensor with shape [N, C, radius * radius, H - (radius - 1), W - (radius - 1)]
	"""
	# pad to ensure the following slice operation is valid
	#pad_beg = int(radius // 2)
	#pad_end = radius - pad_beg

	# the original height and width
	label_shape = labels_4D.size()
	h, w = label_shape[2], label_shape[3]
	new_h, new_w = h - (radius - 1), w - (radius - 1)
	# https://pytorch.org/docs/stable/nn.html?highlight=f%20pad#torch.nn.functional.pad
	#padding = (pad_beg, pad_end, pad_beg, pad_end)
	#labels_4D, probs_4D = F.pad(labels_4D, padding), F.pad(probs_4D, padding)

	# get the neighbors
	la_ns = []
	pr_ns = []
	#for x in range(0, radius, 1):
	for y in range(0, radius, 1):
		for x in range(0, radius, 1):
			la_now = labels_4D[:, :, y:y + new_h, x:x + new_w]
			pr_now = probs_4D[:, :, y:y + new_h, x:x + new_w]
			la_ns.append(la_now)
			pr_ns.append(pr_now)

	if is_combine:
		# for calculating RMI
		pair_ns = la_ns + pr_ns
		p_vectors = torch.stack(pair_ns, dim=2)
		return p_vectors
	else:
		# for other purpose
		la_vectors = torch.stack(la_ns, dim=2)
		pr_vectors = torch.stack(pr_ns, dim=2)
		return la_vectors, pr_vectors


def map_get_pairs_region(labels_4D, probs_4D, radius=3, is_combine=0, num_classeses=21):
	"""get map pairs
	Args:
		labels_4D	:	labels, shape [N, C, H, W].
		probs_4D	:	probabilities, shape [N, C, H, W].
		radius		:	The side length of the square region.
	Return:
		A tensor with shape [N, C, radiu * radius, H // radius, W // raidius]
	"""
	kernel = torch.zeros([num_classeses, 1, radius, radius]).type_as(probs_4D)
	padding = radius // 2
	# get the neighbours
	la_ns = []
	pr_ns = []
	for y in range(0, radius, 1):
		for x in range(0, radius, 1):
			kernel_now = kernel.clone()
			kernel_now[:, :, y, x] = 1.0
			la_now = F.conv2d(labels_4D, kernel_now, stride=radius, padding=padding, groups=num_classeses)
			pr_now = F.conv2d(probs_4D, kernel_now, stride=radius, padding=padding, groups=num_classeses)
			la_ns.append(la_now)
			pr_ns.append(pr_now)

	if is_combine:
		# for calculating RMI
		pair_ns = la_ns + pr_ns
		p_vectors = torch.stack(pair_ns, dim=2)
		return p_vectors
	else:
		# for other purpose
		la_vectors = torch.stack(la_ns, dim=2)
		pr_vectors = torch.stack(pr_ns, dim=2)
		return la_vectors, pr_vectors
	return


def log_det_by_cholesky(matrix):
	"""
	Args:
		matrix: matrix must be a positive define matrix.
				shape [N, C, D, D].
	Ref:
		https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/linalg/linalg_impl.py
	"""
	# This uses the property that the log det(A) = 2 * sum(log(real(diag(C))))
	# where C is the cholesky decomposition of A.
	chol = torch.cholesky(matrix)
	#return 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-6), dim=-1)
	return 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)


def batch_cholesky_inverse(matrix):
	"""
	Args: 	matrix, 4-D tensor, [N, C, M, M].
			matrix must be a symmetric positive define matrix.
	"""
	chol_low = torch.cholesky(matrix, upper=False)
	chol_low_inv = batch_low_tri_inv(chol_low)
	return torch.matmul(chol_low_inv.transpose(-2, -1), chol_low_inv)


def batch_low_tri_inv(L):
	"""
	Batched inverse of lower triangular matrices
	Args:
		L :	a lower triangular matrix
	Ref:
		https://www.pugetsystems.com/labs/hpc/PyTorch-for-Scientific-Computing
	"""
	n = L.shape[-1]
	invL = torch.zeros_like(L)
	for j in range(0, n):
		invL[..., j, j] = 1.0 / L[..., j, j]
		for i in range(j + 1, n):
			S = 0.0
			for k in range(0, i + 1):
				S = S - L[..., i, k] * invL[..., k, j].clone()
			invL[..., i, j] = S / L[..., i, i]
	return invL


def log_det_by_cholesky_test():
	"""
	test for function log_det_by_cholesky()
	"""
	a = torch.randn(1, 4, 4)
	a = torch.matmul(a, a.transpose(2, 1))
	print(a)
	res_1 = torch.logdet(torch.squeeze(a))
	res_2 = log_det_by_cholesky(a)
	print(res_1, res_2)


def batch_inv_test():
	"""
	test for function batch_cholesky_inverse()
	"""
	a = torch.randn(1, 1, 4, 4)
	a = torch.matmul(a, a.transpose(-2, -1))
	print(a)
	res_1 = torch.inverse(a)
	res_2 = batch_cholesky_inverse(a)
	print(res_1, '\n', res_2)


def mean_var_test():
	x = torch.randn(3, 4)
	y = torch.randn(3, 4)

	x_mean = x.mean(dim=1, keepdim=True)
	x_sum = x.sum(dim=1, keepdim=True) / 2.0
	y_mean = y.mean(dim=1, keepdim=True)
	y_sum = y.sum(dim=1, keepdim=True) / 2.0

	x_var_1 = torch.matmul(x - x_mean, (x - x_mean).t())
	x_var_2 = torch.matmul(x, x.t()) - torch.matmul(x_sum, x_sum.t())
	xy_cov = torch.matmul(x - x_mean, (y - y_mean).t())
	xy_cov_1 = torch.matmul(x, y.t()) - x_sum.matmul(y_sum.t())

	print(x_var_1)
	print(x_var_2)

	print(xy_cov, '\n', xy_cov_1)


if __name__ == '__main__':
	batch_inv_test()
