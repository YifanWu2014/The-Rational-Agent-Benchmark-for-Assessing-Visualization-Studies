
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from scipy.stats import norm

'''

percentage_calib = [1.0, 0.5104074960847461, 0.5398135423547441, 0.40907068451088036, 0.44388382715768476, 0.349718670260451, 0.3441832790025844, 0.6878616760170772, 0.6947001319300486]
percentage_behav = [1.0, -0.5030696938210552, -0.445949345389121, -0.9899787991614077, -0.8409955713512278, -1.073724820724339, -1.0248305660291506, 0.19783498024666551, 0.13293434168510126]
percentage_heuristic_low = [1.0, -3.1583520721108322, -3.1583520721108322, -3.1583520721108322, -3.1583520721108322, -3.1583520721108322, -3.1583520721108322, -3.1583520721108322, -3.1583520721108322]
percentage_heuristic_high = [1.0, -2.2298532996428855, -2.2298532996428855, -2.2298532996428855, -2.2298532996428855, -2.2298532996428855, -2.2298532996428855, -2.2298532996428855, -2.2298532996428855]
'''
n_data = 0
#bin_size = 0.05
eps = 1e-5

rand_size = 160 * 2
#prior_freq = 0.7518749999999998


n_ground_truth = 0
condition_dict = {
	"densities":0,
	"intervals":1, 
	"HOPs":2, 
	"QDPs":3
}

means_dict = {
	"FALSE": 0, 
	"TRUE": 1
	}
'''
lo_ground_truth_dict = {
	"0.200670695462151": 0,
	"0.451517615553149": 1,
	"0.699153204620157": 2,
	"0.949427355463583": 3
}
'''
lo_ground_truth_dict = {}

trial_groundtruth_dict = {}
trial_means_dict = {}
trial_condition_dict = {}


sd_diff_dict = {}
n_sd_diff = 0

#lo_ground_truth = np.zeros(8, dtype = float)
lo_ground_truth = []
means = ["FALSE", "TRUE"]
condition = ["densities",
	"intervals", 
	"HOPs", 
	"QDPs"]

decision_truth_map = {}

def inv_log(a):
	return 1.00 / (1.00 + math.exp(0.00 - float(a)))

def qlogis(a):
	return math.log(a / (1.00 - a))

def heuristic(ground_truth, sd_diff):
	return 0.5 + float(ground_truth) * sd_diff / 225.0



 ## condition, means, groundtruth

behavioral_bins = np.arange(100, dtype = float)
#calib_posterior = np.zeros((4, 2, 100), dtype=float)

freq_heuristic = np.zeros((2, 4, 2, 8, 100), dtype=float)

ground_truth_data = []
pred_data = []
means_data = []
condition_data = []



with open('prob_superiority_yifan.csv', newline='') as f:
	data = csv.reader(f)
	headers = next(data) 
	for row in data:
		if not (row[0] in lo_ground_truth_dict):
			#read ground truth pos row[0] in logit space
			lo_ground_truth_dict[row[0]] = n_ground_truth
			# compute probability of winning: first inv_log, then compute p[winning] from pos
			lo_ground_truth.append(norm.cdf(math.sqrt(2) * norm.ppf(inv_log(row[0]))))
			#lo_ground_truth.append(inv_log(row[0]))
			#lo_ground_truth[n_ground_truth] = inv_log(row[0])
			#decision_truth_map[lo_ground_truth[n_ground_truth]] = norm.cdf(math.sqrt(2) * norm.ppf(lo_ground_truth[n_ground_truth]))
			n_ground_truth += 1
		'''
		if not ((row[1], row[3], row[4]) in trial_groundtruth_dict):
			#print(row[4], lo_ground_truth_dict[row[0]], row[0])
			trial_groundtruth_dict[(row[1], row[3], row[4])] = lo_ground_truth_dict[row[0]]
		'''
		ground_truth_data.append(lo_ground_truth_dict[row[0]])
		pred_data.append(row[10])
		means_data.append(row[1])
		condition_data.append(row[3])
		n_data += 1		#print (condition_dict[row[3]], means_dict[row[1]], lo_ground_truth_dict[row[0]], pred, freq[condition_dict[row[3]], means_dict[row[1]], lo_ground_truth_dict[row[0]], pred])


#print (trial_groundtruth_dict)


def binning(bin_size):
	freq = np.zeros((4, 2, 8, 100), dtype=float)
	for i in range(n_data):
		pred = int (inv_log( pred_data[i]) / bin_size)
		# counting condition, means, groundtruth, the response numbers, empirical distribution
		freq[condition_dict[condition_data[i]], means_dict[means_data[i]], ground_truth_data[i], pred] += 1
	return freq


bin_size = 0.02
freq = binning(bin_size)
# compute prior - np.sum(freq[0, 0], axis = 1) is the number of trials with each ground truth, lo_ground_truth winning prob
prior_freq = np.inner(np.sum(freq[0, 0], axis = 1), lo_ground_truth) / np.sum(freq[0, 0])

def compute_freq(bin_num, bin_size):
	return (bin_num + 0.5) * bin_size


def eq_err(a, b):
	if abs(a - b) <= eps:
		return True
	return False

def log(a):
	return np.log(a + eps)

def expect_quadratic(report, pos_truth):
	if type(pos_truth) == 'int':
		tmp_ground_truth = [lo_ground_truth[pos_truth]]
	else:
		tmp_ground_truth = [lo_ground_truth[i] for i in pos_truth]
	tmp_ground_truth = np.array(tmp_ground_truth)
	# decision score
	return decision_score(np.less_equal(prior_freq, report), tmp_ground_truth)
	#return (1 - np.less_equal(prior_freq, report)) * 0.5 * 3.17 + (np.less_equal(prior_freq, report)) * (tmp_ground_truth * 3.17 - 1 )
	
	# optimal score
	# return (np.greater_equal(np.full(np.size(prior_freq), 0.5), prior_freq)) * (np.greater_equal(prior_freq, report)) * ((0.00 - 0.5) * (truth - prior_freq) / (1.00 - prior_freq) + 0.5) + (np.greater_equal(np.full(np.size(prior_freq), 0.5), prior_freq)) * (1 - np.greater_equal(prior_freq, report)) * ((1.00 - 0.5) * (truth - prior_freq)/ (1.00 - prior_freq) + 0.5) + (1 - np.greater_equal(np.full(np.size(prior_freq), 0.5), prior_freq)) * (np.greater_equal(prior_freq, report)) * ((1.00 - 0.5) * (truth - prior_freq) / (0.00 - prior_freq) + 0.5) + (1 - np.greater_equal(np.full(np.size(prior_freq), 0.5), prior_freq)) * (1 - np.greater_equal(prior_freq, report)) * ((0.00 - 0.5) * (truth - prior_freq) / (0.00 - prior_freq) + 0.5)
	
	
	#log score
	#return 0.00 - truth * log(report) + (1.00 - truth) * log(1.00 - report)
	
	#quadratic score
	#return 1 - (truth * (1.00 - report) * (1.00 - report) + (1.00 - truth) * report * report)

def decision_score(decision, truth):
	#return prob * expect_quadratic(1.00, truth) + (1.00 - prob) * expect_quadratic(0.00, truth)
	return decision * (3.17 * truth - 1.00) + (1 - decision) * 3.17 * 0.5
	
	


def rand_draw(freq):
	lst = list(range(0,100))
	sample = random.choices(lst, weights = freq, k = rand_size)
	freq_tmp = np.zeros((1, 100))
	for i in sample:
		freq_tmp[0, i] += 1
	return freq_tmp

def calc_score(freq_tmp, bin_size):
	behavioral_score = 0.0
	for i in range(n_ground_truth):
		#print(freq_tmp[i], compute_freq(behavioral_bins, bin_size), lo_ground_truth[i], expect_quadratic(compute_freq(behavioral_bins, bin_size), lo_ground_truth[i]), np.inner(freq_tmp[i], expect_quadratic(compute_freq(behavioral_bins, bin_size), lo_ground_truth[i])))
		behavioral_score += np.inner(freq_tmp[i], expect_quadratic(compute_freq(behavioral_bins, bin_size), [i])) / np.sum(freq_tmp)
		#print(behavioral_score)
	#print(behavioral_score)

	cond_posterior = np.zeros((n_ground_truth, 100), dtype = float)
	for i in range(n_ground_truth):
		for j in range(100):
			if eq_err( freq_tmp[i, j], 0.0)!= True:
				cond_posterior[i, j] = freq_tmp[i, j] / np.sum(freq_tmp[:, j])
	calib_posterior = np.matmul(lo_ground_truth, cond_posterior)
	#calib_posterior = freq[t_ind, m_ind] / np.sum(freq[t_ind, m_ind], axis = 0)
	calib_behav_score = 0.00
	for i in range(n_ground_truth):
		calib_behav_score += np.inner(freq_tmp[i], expect_quadratic(calib_posterior, [i])) / np.sum(freq_tmp)
	#print(calib_behav_score)
	return behavioral_score, calib_behav_score
	#percentage_calib.append((calib_behav_score - prior_score) / (posterior_score - prior_score))






def main():

	
	prior_score = np.inner(np.sum(freq[0, 0], axis = 1), expect_quadratic(prior_freq, list(range(0,n_ground_truth)))) / np.sum(freq[0, 0])
	print("prior freq", prior_freq)
	print ("prior score: ", prior_score)

	posterior_score = np.inner(np.sum(freq[0, 0], axis = 1), expect_quadratic(lo_ground_truth, list(range(0,n_ground_truth)))) / np.sum(np.sum(freq[0, 0]))
	print(lo_ground_truth)
	print (expect_quadratic(lo_ground_truth, list(range(0,n_ground_truth))))
	print("posterior score: ", posterior_score)
	
	score_ind = []
	score_behav = []
	score_calib = []


	
	decision_ground_truth_dict = {}
	decision_ground_truth_list = []
	decision_ground_truth_count = 0
	decision_freq = np.zeros((4, 2, 8, 2), dtype = float)

	with open('posterior_predictive_draws_decisions_kale2020.csv', newline = '') as f:
		data = csv.reader(f)
		headers = next(data)
		for row in data:
			if not row[0] in decision_ground_truth_dict:
				tmp_ground_truth = inv_log(float(row[0]) + qlogis(0.5 + 1.0 / 3.17))
				flag = True
				for i in range(len(decision_ground_truth_list)):
					t = decision_ground_truth_list[i]
					if eq_err(t, tmp_ground_truth):
						flag = False
						decision_ground_truth_dict[row[0]] = i
				if flag:
					decision_ground_truth_list.append(tmp_ground_truth)
					decision_ground_truth_dict[row[0]] = decision_ground_truth_count
					decision_ground_truth_count += 1
			decision_freq[condition_dict[row[3]], means_dict[row[1]], decision_ground_truth_dict[row[0]], int(row[10])] += 1

	score_decision = []

	for t in range(4):
		for m in range(2):
			tmp_decision = []
			for i in range(100):
				score_tmp = 0.0
				for j in range(0, decision_ground_truth_count):
					sample = random.choices([0, 1], weights = decision_freq[t, m, j], k = rand_size)
					freq_tmp = float(sum(sample)) / rand_size
					if t == 0 and m == 0 and i == 0:
						print(freq_tmp)
					score_tmp += decision_score(freq_tmp, decision_ground_truth_list[j])
				score_tmp = score_tmp / decision_ground_truth_count
				tmp_decision.append(score_tmp)
				if t == 0 and m == 0 and i == 0:
					print(score_tmp)
			tmp_decision.sort()
			tmp_decision = tmp_decision[2:97]
			score_decision.extend(tmp_decision)


	score_bounds = [prior_score, posterior_score]
	score_bounds_ind = [0, 0]

	for t in range(4):
		for m in range(2):
			tmp_behav = []
			tmp_calib = []
			for i in range(100):
				freq_tmp = rand_draw(freq[t, m, 0])
				for j in range(1, n_ground_truth):
					freq_tmp = np.append(freq_tmp, rand_draw(freq[t, m, j]), axis = 0)
				behav_tmp, calib_tmp = calc_score(freq_tmp, bin_size)
				tmp_behav.append(behav_tmp)
				tmp_calib.append(calib_tmp)
			tmp_behav.sort()
			tmp_behav = tmp_behav[2:97]

			tmp_calib.sort()
			tmp_calib = tmp_calib[2:97]

			score_ind.extend([t * 4 + m + 1] * 95)
			score_behav.extend(tmp_behav)
			score_calib.extend(tmp_calib)

	plt.scatter(score_bounds_ind, score_bounds, s = 100)
	plt.scatter(score_ind, score_calib, s = 1)
	plt.scatter(score_ind, score_behav, s = 1)
	plt.scatter(score_ind, score_decision, s = 2)
	#plt.scatter(decision_score_ind, decision_score_final, s = 50)
	#plt.scatter(decision_score_ind, decision_score_final_flip, s=30)
	plt.show()


'''
	for t in range(4):
		for m in range(2):
			plt.subplot(8, 1, t * 2 + m  + 1)
			plt.plot(bias_ind, decision_wrong_cnt[t, m] / decision_pred_cnt[t, m])
	plt.show()
'''

main()
