import sys
import nltk
# nltk.download('punkt')   # Only uncomment this if 'punkt' is not downloaded
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

trainable = False

try:
    with open('intents.json') as file:
        data = json.load(file)
except FileNotFoundError as e:
    print("File doesn't exist: \n{}".format(e))
    sys.exit(1)

if not trainable:
    try:
        with open('data.pickle', 'rb') as f:
            words, labels, training, output = pickle.load(f)
    except:
        trainable = True

if trainable:
    words = []  # this will contain the root words
    labels = []  # this will contain the tags
    docs_patterns = []  # this will contain each pattern list
    docs_labels = []  # this will contain each tag, but many times to get the amount of the tags

    for intent in data['intents']:
        for pattern in intent['patterns']:
            # Equals the list pattern to words_pattern without punctuation marks
            # ['Hi']
            # ['How', 'are', 'you']
            # ...
            words_pattern = nltk.regexp_tokenize(pattern, "(\d+|\w+)")

            # Extend the list words with the list 'words_pattern'
            # words = ['Hi', 'How', 'are', 'you', 'Is', 'anyone', 'there', ...]
            words.extend(words_pattern)

            # Append the list 'words_pattern' to the list doc_x (not extending)
            # doc_patterns = [['Hi'], ['How', 'are', 'you'], ['Is', 'anyone', 'there']], ... ]
            docs_patterns.append(words_pattern)

            # Adding the tags to the list doc_y
            # doc_labels = ['greeting', 'greeting', 'greeting', 'greeting', 'greeting', 'goodbye', 'goodbye', ...]
            docs_labels.append(intent["tag"])

        # Append each label one time in list labels
        # labels = ['greeting', 'goodbye', 'thanks', ...]
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    # Stemmer determines the root word -> "happening" becomes "happen"
    # The list words is now the list with root words
    words = [stemmer.stem(word.lower()) for word in words]

    # Sorting the list words and make each root word unique in it
    # words = ['acceiv', 'ag', 'anyon', 'ar', 'bye', 'card', 'cash', 'credit',...]
    words = sorted(list(set(words)))

    # Sorting of list labels
    # labels = ['age', 'goodbye', 'greeting', 'hours', 'name', 'opentoday', ...]
    labels = sorted(labels)
    print(labels)

    training = []
    output = []

    # out_empty = [0, 0, 0, 0, 0, ...]
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_patterns):
        bag = []

        # Convert each word to root word
        root_words_from_doc = [stemmer.stem(word.lower()) for word in doc]

        for word in words:
            if word in root_words_from_doc:  # Represent words as 0 and 1 in the bag
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_labels[x])] = 1  # Represent labels as 0 and 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open('data.pickle', 'wb') as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

if not trainable:
    try:
        model.load("model.tflearn")
    except:
        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        model.save("model.tflearn")
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(input_user, words):
    bag = [0 for _ in range(len(words))]

    user_input_words = nltk.regexp_tokenize(input_user, "(\d+|\w+)")
    user_input_words = [stemmer.stem(word.lower()) for word in user_input_words]

    for user_input_word in user_input_words:
        for i, word in enumerate(words):
            if word == user_input_word:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        try:
            input_user = input("You: ")
            if input_user.lower() == "quit":
                break

            results = model.predict([bag_of_words(input_user, words)])[0]
            results_index = numpy.argmax(results)
            tag = labels[results_index]

            if results[results_index] > 0.7:
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                print(random.choice(responses))
            else:
                print("I am sorry, I didn't understand you")

        except UnboundLocalError as e:
            print("Some error occurred: {}".format(e))


chat()

# TechWithTim: Python AI chatbot tutorial part 1-4
# https://techwithtim.net/tutorials/ai-chatbot/
# Geraadpleegd op 8 novemeber 2019
