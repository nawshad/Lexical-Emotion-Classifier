'''
Written by Nawshad Farruque
@param1: filename
@param2: threshold
usage: python info.py <filename> <threshold>
'''

#create a key(word)=>value(emotion labels) dictionary, which will be our lexicon.
import numpy as np
import sys
from sklearn.model_selection import LeaveOneOut

def CBET_dataloader(filename):
	tweets = []
	labels = []
	with open (filename, 'r') as file:
		for line in file:
			line_content = line.strip("\n").split("\t\t")
			tweets.append(line_content[0])
			labels.append(line_content[1])
	return tweets,labels

def lexicon_builder(tweets, labels, no_of_emotions):
	lexicon = dict()
	#with open (filename, 'r') as file:
	for i in range(len(tweets)):
		#line_content = line.strip("\n").split("\t\t")
		#get the uniq words in each tweet
		#uniq_words = set(line_content[0].lower().split())
		#not taking set as ameneh did not consider uniq words in each tweets
		uniq_words = tweets[i].lower().split()
		#print(uniq_words)
		#check if the key exist, if not, add to the lexicon 
		#with np.array with 9 zeros, else, update corresponding 
		#values of a key
		for word in uniq_words:
			if word not in lexicon:
				lexicon[word] = np.zeros((no_of_emotions))
				# update with one which is appeared for specific emotion
				lexicon[word][int(labels[i])] = 1
			else:
				lexicon[word][int(labels[i])] += 1.0
	return lexicon

def test():
	a = np.array([1,2,2,1])
	print(np.where(a == max(a))[0][0])

def lexical_classifier(tweet, lexicon, emotions):
	init_value  = np.zeros(len(emotions))
	for word in tweet.split():
		if word in lexicon:
			init_value += lexicon[word]
	#print(np.where(init_value == max(init_value))[0][0])
	pred_label = np.where(init_value == max(init_value))[0][0]
	return pred_label

# utilty function
def get_smaller_dataset(filename, range):
	tweets_and_labels = []
	
	with open (filename, 'r') as file:
		count = 0
		covered_ids = []
		for line in file:
			line_content = line.strip("\n").split("\t\t")
			if count <= range: 
				if line_content[1] not in covered_ids:
					tweets_and_labels.append(line_content)
					count += 1
			else:
				covered_ids.append(line_content[1])
				count = 0
				
	with open('CBET-single-small.txt','w') as file:
		for item in tweets_and_labels:
			file.write(item[0]+"\t\t"+item[1]+"\n")

def inf_vocab_builder(tweets, labels, emotions, threshold):
	X = np.array(tweets)
	y = np.array(labels)
	loo = LeaveOneOut()
	loo.get_n_splits(X)
	#print(loo)
	info_vocab = dict()
	correct_count = 0
	total = 0
	for train_index, test_index in loo.split(X):
		print("Done with TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		lexicon = lexicon_builder(X_train, y_train, len(emotions))
		pred_label = lexical_classifier(X_test[0], lexicon, emotions)
		#print(X_test[0]+" Gold label: "+str(y_test[0])+" Predicted label: "+str(pred_label))
		#print("pred_label type", type(pred_label))
		#print("gold_label type", type(int(y_test[0])))
		if pred_label == int(y_test[0]):
			correct_count += 1
		#print(str(correct_count)+"/"+str(total)+" are correct so far")
		for word in X_test[0].split():
			if word not in info_vocab:
				info_vocab[word] = np.zeros((2)) #two items: 0 contain correctly classify, 1 contain total classify
				# look if this word is used for classification and if used for correct classification update corresponding bits
				if word in lexicon:
					info_vocab[word][1] = 1
					if pred_label == int(y_test[0]):
						info_vocab[word][0] = 1
			else:
				if word in lexicon:
					info_vocab[word][1] += 1
					if pred_label == int(y_test[0]):
						info_vocab[word][0] += 1
		total += 1
		
	vocab_list = []
	for key, value in info_vocab.items():
		if value[1] > 0:
			if value[0]/value[1] >= threshold:
				vocab_list.append(key)

	print("Vocabulary created!")
	print("info_vocab size:", len(info_vocab))
	print('final vocab size:', len(vocab_list))

	with open('inf_vocab.txt', 'w') as file:
		for item in vocab_list:
			file.write(item+'\n')
		with open('emos.txt', 'r') as emos:
			for line in emos:
				file.write(line)
	
	print("Vocabulary+emoticons written!")

def lexical_classifier_word(word, lexicon, emotions):
	init_value  = np.zeros(len(emotions))
	#for word in tweet.split():
	if word in lexicon:
		init_value = lexicon[word]
	#print(np.where(init_value == max(init_value))[0][0])
	pred_label = np.where(init_value == max(init_value))[0][0]
	return pred_label


def inf_vocab_builder_ameneh(tweets, labels, emotions, threshold):
	X = np.array(tweets)
	y = np.array(labels)
	loo = LeaveOneOut()
	loo.get_n_splits(X)
	#print(loo)
	info_vocab = dict()
	correct_count = 0
	total = 0
	for train_index, test_index in loo.split(X):
		print("Done with TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		lexicon = lexicon_builder(X_train, y_train, len(emotions))
		pred_label = lexical_classifier_word(X_test[0], lexicon, emotions)
		#print(X_test[0]+" Gold label: "+str(y_test[0])+" Predicted label: "+str(pred_label))
		#print("pred_label type", type(pred_label))
		#print("gold_label type", type(int(y_test[0])))
		if pred_label == int(y_test[0]):
			correct_count += 1
		#print(str(correct_count)+"/"+str(total)+" are correct so far")
		for word in X_test[0].split():
			if word not in info_vocab:
				info_vocab[word] = np.zeros((2)) #two items: 0 contain correctly classify, 1 contain total classify
				# look if this word is used for classification and if used for correct classification update corresponding bits
				if word in lexicon:
					#print(info_vocab[word])
					info_vocab[word][1] = np.sum(lexicon[word])
					if pred_label == int(y_test[0]):
						info_vocab[word][0] = 1

			else:
				if word in lexicon:
					#info_vocab[word][1] += 1
					if pred_label == int(y_test[0]):
						info_vocab[word][0] += 1
		total += 1
	
	vocab_list = []
	
	for key, value in info_vocab.items():
		if value[1] > 0:
			if value[0]/value[1] >= threshold:
				vocab_list.append(key)

	print("Vocabulary created!")
	print("info_vocab size:", len(info_vocab))
	print('final vocab size:', len(vocab_list))

	with open('inf_vocab.txt', 'w') as file:
		for item in vocab_list:
			file.write(item+'\n')
		with open('emos.txt', 'r') as emos:
			for line in emos:
				file.write(line)
	
	print("Vocabulary+emoticons written!")



def main():
	emotions = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise', 'thankfulness', 'disgust', 'guilt']
	tweets, labels = CBET_dataloader(sys.argv[1]) #all tweets are made lowercase here
	threshold = float(sys.argv[2])
	#print(lexicon_builder(tweets, labels, len(emotions))['happier'])
	inf_vocab_builder(tweets, labels, emotions, threshold)


	#get_smaller_dataset('CBET-single.txt', 100)

if __name__ == '__main__':
	main()
