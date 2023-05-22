from flask import Flask, render_template, request
  
app = Flask(__name__)

from pickle import load
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model
# from nltk.translate.bleu_score import corpus_bleu

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

def encode_sequences_input(tokenizer, length, lines):
	line = [lines]
	# integer encode sequences
	X = tokenizer.texts_to_sequences(line)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate target given source sequence
def predict_sequence(model, tokenizer, source):
	print(source)
	prediction = model.predict(source, verbose=0)[0]
	integers = [argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)

# evaluate the skill of the model
def evaluate_model_input(model, tokenizer, source, raw_dataset, lang):
	for i, text in enumerate(source):
		# translate encoded source text
		text = text.reshape((1, text.shape[0]))
		if lang == "e":
			translation = predict_sequence(model, ger_tokenizeretog, text)
		elif lang == "g":
			translation = predict_sequence(model, eng_tokenizergtoe, text)
		return translation
	# translate encoded source text
	# source = source.reshape((1, enumerate(source).shape[0]))
	

# english to german
# load datasets
datasetetog = load_clean_sentences('etog-both.pkl')
trainetog = load_clean_sentences('etog-train.pkl')
testetog = load_clean_sentences('etog-test.pkl')
# prepare english tokenizer
eng_tokenizeretog = create_tokenizer(datasetetog[:, 0])
eng_vocab_sizeetog = len(eng_tokenizeretog.word_index) + 1
eng_lengthetog = max_length(datasetetog[:, 0])
# prepare german tokenizer
ger_tokenizeretog = create_tokenizer(datasetetog[:, 1])
ger_vocab_sizeetog = len(ger_tokenizeretog.word_index) + 1
ger_lengthetog = max_length(datasetetog[:, 1])
# prepare data
trainXetog = encode_sequences(eng_tokenizeretog, eng_lengthetog, trainetog[:, 1])
testXetog = encode_sequences(ger_tokenizeretog, ger_lengthetog, testetog[:, 1])


# german to english
# load datasets
datasetgtoe = load_clean_sentences('gtoe-both.pkl')
traingtoe = load_clean_sentences('gtoe-train.pkl')
testgtoe = load_clean_sentences('gtoe-test.pkl')
# prepare english tokenizer
eng_tokenizergtoe = create_tokenizer(datasetgtoe[:, 0])
eng_vocab_sizegtoe = len(eng_tokenizergtoe.word_index) + 1
eng_lengthgtoe = max_length(datasetgtoe[:, 0])
# prepare german tokenizer
ger_tokenizergtoe = create_tokenizer(datasetgtoe[:, 1])
ger_vocab_sizegtoe = len(ger_tokenizergtoe.word_index) + 1
ger_lengthgtoe = max_length(datasetgtoe[:, 1])
# prepare data
trainXetog = encode_sequences(eng_tokenizergtoe, eng_lengthgtoe, traingtoe[:, 1])
testXetog = encode_sequences(ger_tokenizergtoe, ger_lengthgtoe, testgtoe[:, 1])

# load model
modeletog = load_model('modeletog.h5')
modelgtoe = load_model('modelgtoe.h5')

# language = input("which is the source language? (e/g): ")
# if language == "e":
    # toTranslate = input("enter a phrase: ")
    # toTranslateX = encode_sequences_input(eng_tokenizeretog, eng_lengthetog, toTranslate)
    # evaluate_model_input(modeletog, ger_tokenizeretog, toTranslateX, trainetog[:, 1], language)
# elif language == "g":
#     toTranslate = input("enter a phrase: ")
#     toTranslateX = encode_sequences_input(ger_tokenizergtoe, ger_lengthgtoe, toTranslate)
#     evaluate_model_input(modelgtoe, eng_tokenizergtoe, toTranslateX, traingtoe[:, 1], language)





@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        toTranslate = request.form.getlist("translationinput")[0]
        language = request.form.getlist("language")[0]
        if language == "e": 
            currenttrans = "currently translating: english to german"
            toTranslateX = encode_sequences_input(eng_tokenizeretog, eng_lengthetog, toTranslate)
            translation = evaluate_model_input(modeletog, ger_tokenizeretog, toTranslateX, trainetog[:, 1], language)
        elif language == "g": 
            currenttrans = "currently translating: german to english"
            toTranslateX = encode_sequences_input(ger_tokenizergtoe, ger_lengthgtoe, toTranslate)
            translation = evaluate_model_input(modelgtoe, eng_tokenizergtoe, toTranslateX, traingtoe[:, 1], language)
        print("translation: " + translation)
        print("to translate: " + toTranslate)
        return render_template('translate.html', translation=translation, needstranslating=toTranslate, currenttrans=currenttrans, lang=language)
    return render_template('home.html')

if __name__ == '__main__':
    app.run()