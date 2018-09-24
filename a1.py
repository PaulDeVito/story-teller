import numpy as np
import math
import scipy as scp
import scipy.spatial as spat
from random import shuffle

def count(filename):
	file = open(filename, "r")

	tokens = {}
	bigrams = {}

	# counting
	corpus = file.read().split()
	prev_word = "<s>"
	tokens["<s>"] = 1
	tokens["</s>"] = 1
	for w in range(len(corpus)):
		word = corpus[w]

		if is_word(word):

			# token counting
			if word in tokens:
				tokens[word] += 1
			else:
				tokens[word] = 1

			# bigram counting
			bigram = prev_word + " " + word
			if bigram in bigrams:
				bigrams[bigram] += 1
			else:
				bigrams[bigram] = 1

			# final
			if w == len(corpus) - 1:
				next_word = "</s>"
				next_bigram = word + " " + next_word
				bigrams[next_bigram] = 1

			prev_word = word

		if word == ".":
			bigram = prev_word + " </s>"
			tokens["</s>"] += 1
			if bigram in bigrams:
				bigrams[bigram] += 1
			else:
				bigrams[bigram] = 1
			prev_word = "<s>"
			tokens["<s>"] += 1

	return tokens, bigrams


def is_word(word):
	for i in range(len(word)):
		if ord(word[i]) != 39 and (ord(word[i]) >= 123 or ord(word[i]) <= 96):
			return False
	return True


def probabilities(tokens, bigrams, smooth=False):
	probs = {}
	total_tokens = sum(tokens.values()) - tokens['<s>'] - tokens['</s>']
	for bigram in bigrams:
		token, next_token = bigram.split()[0], bigram.split()[1]
		if token not in probs:
			probs[token] = {"prob": 0, "next": {}}
			if smooth:
				probs[token]["next"]['<UNK>'] = 0.000027 / tokens[token]
		probs[token]["prob"] = float(tokens[token]) / total_tokens
		probs[token]["next"][next_token] = float(bigrams[bigram]) / tokens[token]
	return probs


def random_sentences(file_name, unigram=True, fragment=None):
	tokens, bigrams = count(file_name)

	probs = probabilities(tokens, bigrams)

	sentence = '<s>'
	prev = '<s>'

	if fragment:
		sentence = fragment
		prev = fragment.split()[-1]

	if unigram:
		token_keys = probs.keys()
		token_probs = probs.values()
		token_probs = [x['prob'] for x in token_probs]
		for x in range(15):
			sample = np.random.choice(token_keys, 1, token_probs)
			sentence = sentence + " " + sample[0]
	else:
		end = False
		while not end:
			valid_keys = probs[prev]['next'].keys()
			valid_probs = probs[prev]['next'].values()
			sample = np.random.choice(valid_keys, 1, valid_probs)
			sentence = sentence + " " + sample[0]
			prev = sample[0]
			if prev == '</s>':
				end = True
	return sentence


def smooth(file_name):
	tokens, bigrams = count(file_name)

	for t in tokens:
		if tokens[t]>=2:
			tokens[t] = tokens[t]-0.75
		elif tokens[t] == 1:
			tokens[t] = tokens[t]-0.5
	tokens['<UNK>'] = 0.000027

	for b in bigrams:
		if bigrams[b]>=2:
			bigrams[b] = bigrams[b]-0.75
		elif bigrams[b] == 1:
			bigrams[b] = bigrams[b]-0.5
	bigrams['<UNK> <UNK>'] = 0.000027

	return tokens, bigrams


def perplexity(development, train):
	N = (open(development, "r")).read().split() + ["</s>"]
	tokens, bigrams = smooth(train)
	probs = probabilities(tokens, bigrams, smooth=True)
	unknown_prob = probs["<UNK>"]["next"]["<UNK>"]               # TODO: shouldn't this depend on the the preceeding word?

	pp = 0
	prev_word = "<s>"
	for x in range(len(N)):
		word = N[x]
		bigram = prev_word + " " + word
		if bigram in bigrams:
			p = -math.log(probs[prev_word]["next"][word])
		else:
			p = -math.log(unknown_prob)
		prev_word = word
		pp += p
	pp = math.exp(pp/(len(N)-1))

	return pp


def get_sequence_probability(sequence, model):
	p_total = 1
	prev_word = "<s>"
	for word in sequence:
		bigram = prev_word + " " + word
		if prev_word in model:
			if word in model[prev_word]["next"]:
				p_total = p_total * model[prev_word]["next"][word]
			else:
				p_total = p_total * model[prev_word]["next"]['<UNK>']
		else:
			p_total = p_total * model["<UNK>"]["next"]["<UNK>"]

	return p_total


def make_prediction(line, model_obama, model_trump):
	words = line.split()
	words = words + ["</s>"]

	p_obama = get_sequence_probability(words, model_obama)
	p_trump = get_sequence_probability(words, model_trump)

	if (p_obama >= p_trump):
		return 0
	else:
		return 1


def classify(train_obama, train_trump, dev_obama, dev_trump, test):
	unigram_counts, bigram_counts = smooth(train_obama)
	model_obama = probabilities(unigram_counts, bigram_counts, True)
	unigram_counts, bigram_counts = smooth(train_trump)
	model_trump = probabilities(unigram_counts, bigram_counts, True)
	predictions = {}
	
	line_id = 0
	sequences = open(test, "r").readlines()
	for line in sequences:
		predictions[line_id] = make_prediction(line, model_obama, model_trump)
		line_id += 1

	return predictions
		
def output_predictions(predictions):
	file = open("output.txt", "w")
	file.write("Id,Prediction\n")

	for i in range(len(predictions.items())):
		file.write(str(i) + "," + str(predictions[i]) + "\n")


def build_vectors(filename):
	lines = open(filename, "r").readlines()
	embeddings = {}
	dimensions = 0

	for line in lines:
		line = line.split()
		word = line[0]
		if not is_word(word):
			continue
		v = np.array(line[1:]).astype(np.float)
		embeddings[word] = v
		dimensions = v.shape[0]


	return (embeddings, dimensions)


def find_analogy(a, b, c, embeddings):
	max_dist = 0
	analogy = ""
	v = np.add(np.subtract(embeddings[b],embeddings[a]),embeddings[c])
	for word, d in embeddings.items():
		if word == c:
			continue
		cos = np.dot(d, v) / (np.linalg.norm(d) * np.linalg.norm(v))
		if cos > max_dist:
			max_dist = cos
			analogy = word

	return analogy


# n is the number of analogies to test
def evaluate_embedding(embeddings, test):
	lines = open(test, "r").readlines()
	vector_map, d = build_vectors(embeddings)
	n = len(lines)
	m = len(vector_map)
	print(n,m,d)

	a = np.empty((n,d))
	b = np.empty((n,d))
	c = np.empty((n,d))

	answer_key = []
	prompt_key = []
	i = 0
	for line in lines:
		line = line.split()
		try:
			a[i] = vector_map[line[0]]
			b[i] = vector_map[line[1]]
			c[i] = vector_map[line[2]]
		except:
			a[i] = np.ones(d)
			b[i] = np.ones(d)
			c[i] = np.ones(d)

		prompt_key.append(line[2])
		answer_key.append(line[3])
		i += 1

	analogy_vectors = np.add(np.subtract(b,a),c)
	word_vectors = np.empty((m,d))

	words = []
	i = 0
	for w, v in vector_map.items():
		word_vectors[i] = v
		words.append(w)
		i += 1

	word_lengths = np.linalg.norm(word_vectors,axis=1)

	i = 0
	num_correct = 0
	for analogy_vector in analogy_vectors:
		dists = np.dot(analogy_vector, word_vectors.T)
		dists = np.divide(dists, np.linalg.norm(analogy_vector))
		dists = np.divide(dists, word_lengths)

		max_idx = np.argmax(dists)
		prediction = words[max_idx]
		if prediction == prompt_key[i]:
			dists[max_idx] = 0
			max_idx = np.argmax(dists)
			prediction = words[max_idx]
		# print(prediction +", " + answer_key[i])
		if prediction == answer_key[i]:
			num_correct += 1

		i += 1
		if i%100 == 0:
			print("Progress: " + str(int((i/float(n))*100)) + "%,  Accuracy: " + str(int((float(num_correct)/float(i)*100))) + "%")

		
	accuracy = float(num_correct) / float(n)
	print(accuracy)
	return accuracy


	# num_correct = 0
	# i = 0
	# for analogy in analogies:
	# 	if i >= n:
	# 		break
	# 	try:
	# 		closest_word = find_analogy(analogy["a"], analogy["b"], analogy["c"], embeddings)
	# 	except:
	# 		continue
	# 	print(analogy["a"] + " is to " + analogy["b"] + " as " + analogy["c"] + " is to " + closest_word + " (" + analogy["d"] + ")")
	# 	if analogy["d"] == closest_word:
	# 		num_correct += 1
	# 	i += 1

	# return num_correct/n



 
train_obama = "Assignment1_resources/train/obama.txt"
train_trump = "Assignment1_resources/train/trump.txt"
dev_obama = "Assignment1_resources/development/obama.txt"
dev_trump = "Assignment1_resources/development/trump.txt"
test = "Assignment1_resources/test/test.txt"

analogy_test = "Assignment1_resources/analogy_test.txt"
glove_wikipedia_small = "glove6B/glove.6B.50d.txt"
glove_wikipedia_largest = "glove6B/glove.6B.300d.txt"

# section 2
# obama_unigrams, obama_bigrams = count("Assignment1_resources/train/obama.txt")
# obama_probability_model = probabilities(obama_unigrams, obama_bigrams)

# trump_unigrams, trump_bigrams = count("Assignment1_resources/train/trump.txt")
# trump_probability_model = probabilities(trump_unigrams, trump_bigrams)

# section 3
# obama_unigram_random = random_sentences("Assignment1_resources/train/obama.txt")
# obama_bigram_random = random_sentences("Assignment1_resources/train/obama.txt", False)

# trump_unigram_random = random_sentences("Assignment1_resources/train/trump.txt")
# trump_bigram_random = random_sentences("Assignment1_resources/train/trump.txt", False)

# section 4
# obama_unigrams_smooth, obama_bigrams_smooth = smooth("Assignment1_resources/train/obama.txt")
# obama_probability_model_smooth = probabilities(obama_unigrams_smooth, obama_bigrams_smooth)

# trump_unigrams_smooth, trump_bigrams_smooth = smooth("Assignment1_resources/train/trump.txt")
# trump_probability_model_smooth = probabilities(trump_unigrams_smooth, trump_bigrams_smooth)

# section 5
# perplexity = perplexity(dev_obama, train_obama)

#section 6
# predictions = classify(train_obama, train_trump, dev_obama, dev_trump, test)
# output_predictions(predictions)


#section 7
# analogy = find_analogy("man", "woman", "king", build_vectors(glove_wikipedia_small))
# print(analogy)

evaluate_embedding(glove_wikipedia_largest,analogy_test)
