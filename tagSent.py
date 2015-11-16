__author__ = 'supriyaanand'

import re
import xmltodict
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.collocations import *
import random
from nltk.corpus import stopwords
import pprint

class tagSent:
    def __init__(self, dataset_file_name_pos, dataset_file_name_neg):
        self.dataset_file_name_pos = dataset_file_name_pos
        self.dataset_file_name_neg = dataset_file_name_neg
        self.bigrams_pos = []
        self.bigrams_neg = []
        self.positiveEmoticons = re.compile(r'[:;B=]+[-_]*[)PD]+')
        self.negativeEmoticons = re.compile(r'[:;B]+[-_]*[(|]+')
        self.stressedCharacters = re.compile(r'(\w)\1{2,}')
        self.cachedStopWords = stopwords.words("english")

    def loadDataSet(self):
        self.dataset_pos = []
        self.dataset_neg = []
        review = ''
        flag = False
        with open(self.dataset_file_name_pos,'r') as f:
            for line in f:
                if line.startswith('<review>'):
                    flag = True
                if flag:
                    review += line.strip()
                if line.strip().endswith('</review>'):
                    flag = False
                    self.dataset_pos.append(review)
                    review = ''


        with open(self.dataset_file_name_neg,'r') as f:
            for line in f:
                if line.startswith('<review>'):
                    flag = True
                if flag:
                    review += line.strip()
                if line.strip().endswith('</review>'):
                    flag = False
                    self.dataset_neg.append(review)
                    review = ''



    def loadLexicons(self, pos_words_file, neg_words_file):
        self.positive_list_file = pos_words_file
        self.negative_list_file = neg_words_file
        self.pos_words = {}
        self.neg_words = {}
        with open(self.positive_list_file,'r') as f:
            for line in f:
                line = line.strip()
                self.pos_words[line] = 1

        with open(self.negative_list_file,'r') as f:
            for line in f:
                line = line.strip()
                self.neg_words[line] = 1



    def extractReviewAndTag(self):
        self.reviews_pos = []
        self.reviews_neg = []
        self.all_tokens_positive = []
        self.all_tokens_negative = []

        for review in self.dataset_pos:
            xml_review = xmltodict.parse(review)
            id = xml_review['review']['asin']
            review_text = xml_review['review']['review_text']
            self.reviews_pos.append([id, review_text])
            self.all_tokens_positive.extend(self.tokenize(review_text))


        for review  in self.dataset_neg:
            xml_review = xmltodict.parse(review)
            id = xml_review['review']['asin']
            review_text = xml_review['review']['review_text']
            self.reviews_neg.append([id, review_text])
            self.all_tokens_negative.extend(self.tokenize(review_text))

    def isPositiveWord(self, token):
        return self.pos_words.has_key(token)

    def isNegativeWord(self, token):
        return self.neg_words.has_key(token)

    def numberOfPositives(self, tokens):
        count = 0
        for token in tokens:
            if self.isPositiveWord(token):
                count += 1
        return count

    def numberOfNegatives(self, tokens):
        count = 0
        for token in tokens:
            if self.isNegativeWord(token):
                count += 1
        return count

    def tokenize(self, review_text):
        tokenized_data = review_text.split()
        tokenized_data = list(set(tokenized_data) - set(self.cachedStopWords))
        return tokenized_data

    def nGramBuilder(self, tokens, sentTag):
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        trigram_measures = nltk.collocations.TrigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(tokens)
        finder.apply_freq_filter(7)
        if sentTag == "positive":
            self.bigrams_pos.append(finder.nbest(bigram_measures.pmi, 300))
        else:
            self.bigrams_neg.append(finder.nbest(bigram_measures.pmi, 300))

    def frequencyOfPositiveBigrams(self, tokens):
        pass

    def check_bigram_presense(self, word_list, token):
        ret_val = 0
        for bigrams in word_list:
            if token == bigrams[0] or token == bigrams[1]:
                ret_val += 1
        return ret_val

    def numberOfHappyEmoticons(self, review_text):
        return len(self.positiveEmoticons.findall(review_text))

    def numberOfNegativeEmoticons(self, review_text):
        return len(self.negativeEmoticons.findall(review_text))

    def containsStressedCharacters(self, review_text):
        True if len(self.stressedCharacters.findall(review_text)) > 0 else False

    def generateFeatureSet(self, review):
        tokens = self.tokenize(review[1])
        features = {}
        features['numberOfPositives'] = self.numberOfPositives(tokens)
        features['numberOfNegatives'] = self.numberOfNegatives(tokens)
        features['positiveBigramCount'] = 0
        features['negativeBigramCount'] = 0
        for token in tokens:
            features['positiveBigramCount'] += self.check_bigram_presense(self.bigrams_pos, token)
            features['negativeBigramCount'] += self.check_bigram_presense(self.bigrams_neg, token)
        features['numOfPosEmoticons'] = self.numberOfHappyEmoticons(review[1])
        features['numOfNegEmoticons'] = self.numberOfNegativeEmoticons(review[1])
        features['stressedCharacters'] = self.containsStressedCharacters(review[1])
        features['text'] = review[1]
        return features


if __name__ == '__main__':
    sentObj = tagSent('positive_review.txt', 'negative_review.txt')
    sentObj.loadDataSet()
    sentObj.loadLexicons('positive_words_list.txt', 'negative_words_list.txt')
    sentObj.extractReviewAndTag()
    sentObj.nGramBuilder(sentObj.all_tokens_positive, "positive")
    sentObj.nGramBuilder(sentObj.all_tokens_negative, "negative")
    featureSets = []
    featureSets.extend([(sentObj.generateFeatureSet(review), "positive") for review in sentObj.reviews_pos])
    featureSets.extend([(sentObj.generateFeatureSet(review), "negative") for review in sentObj.reviews_neg])
    random.shuffle(featureSets)
    train_set, test_set = featureSets[:1200],featureSets[1201:]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))
    print classifier.show_most_informative_features(150)

    errors = []
    for review in test_set:
        guess = classifier.classify(review[0])
        if guess != review[1]:
            errors.append( (review[1], guess, review[0]['text'], review[0]) )

    for (tag, guess, text, features) in sorted(errors):
        print "guess " + guess + " text " + text + " tag" + tag + " features "
        pprint.pprint(features)


    print(nltk.classify.accuracy(classifier, test_set))
    print classifier.show_most_informative_features(150)