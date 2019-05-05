#Lonely_Chatbot Code

import nltk
import warnings
import numpy as np
import random
import string

warnings.filterwarnings("ignore")

f = open('drug_effects.txt', 'r', errors='ignore')
raw = f.read()
raw = raw.lower()  # converts to lowercase
# nltk.download('punkt') # first-time use only
# nltk.download('wordnet') # first-time use only
sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words

sent_tokens[:2]

word_tokens[:5]

lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
	return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
	return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey", "hola")
GREETING_RESPONSES = ["Hi", "Hey", "Hola", "Hi There", "Hello", "Are you are talking to me?", "Bonjour", "Ciao"]


# Did you start with a greeting?
def greeting(sentence):
	for word in sentence.split():
		if word.lower() in GREETING_INPUTS:
			return random.choice(GREETING_RESPONSES)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Generating response
def response(user_response):
	robo_response = ''
	sent_tokens.append(user_response)
	TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
	tfidf = TfidfVec.fit_transform(sent_tokens)
	vals = cosine_similarity(tfidf[-1], tfidf)
	idx = vals.argsort()[0][-2]
	flat = vals.flatten()
	flat.sort()
	req_tfidf = flat[-2]
	if (req_tfidf == 0):
		robo_response = robo_response + "Sorry! No Comprendo....Try Again!."
		return robo_response
	else:
		robo_response = robo_response + sent_tokens[idx]
		return robo_response


flag = True
print("Hi! My name is CBot. I'm an expert at providing medication side effects. ")
print(" -- ")
print("At anytime, type 'Bye' to end our conversation")
print(" -- ")
print("To begin, enter the name of a medication?")

while (flag == True):
	user_response = input()
	user_response = user_response.lower()
	if (user_response != 'bye'):
		if (user_response == 'thanks' or user_response == 'thank you'):
			flag = False
			print("CBot: You are welcome..")
		else:
			if (greeting(user_response) != None):
				print("CBot: " + greeting(user_response))
			else:
				print("CBot: ", end="")
				print(response(user_response))
				sent_tokens.remove(user_response)
	else:
		flag = False
		print("CBot: Bye! Cut back on medications!..")


