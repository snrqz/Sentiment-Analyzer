import pandas as pd
from collections import Counter


def remove_index(data):
    return data.drop("Unnamed: 0", axis=1)


def data_fixing(data):
    cleaned_data = data[data["text"].str.split().str.len() > 10]
    return remove_index(cleaned_data)


def handle_txt(data):
    return [each for each in data.text]


def handle_x(X):
    return [x for x in X]


def mapping_label(data, y):
    sentiments = ["sad", "joy", "love", "anger", "fear", "surprise"]
    mapping = {_: sentiment for sentiment, _ in zip(sentiments, range(len(data[y].unique())))}
    data[y] = data[y].map(mapping)


def label_distances(X, y, text):
    unique_txt = list(set(word for doc in text for word in doc.split()))
    df_train = pd.DataFrame(X)
    df_train["_label_"] = y
    dic_container = {word: [] for word in unique_txt}
    vocab_size = len(unique_txt)

    for yUnique in df_train["_label_"].unique():
        filtered_data = df_train[df_train["_label_"] == yUnique]
        words_frequency = Counter(" ".join(filtered_data["text"]).split())
        total_words_in_class = sum(words_frequency.values())
        for unique_word in unique_txt:
            frequency = words_frequency.get(unique_word, 0)
            smoothed_prob = (frequency + 1) / (total_words_in_class + vocab_size)
            dic_container[unique_word].append(smoothed_prob)

    thatdf = pd.DataFrame(dic_container)
    thatdf["_label_"] = df_train["_label_"].unique()
    return thatdf


def convert_distances_to_booleans(data, text):
    unique_txt, dic_container = list(set(text)), dict()

    for unique_word in unique_txt:
        lcontainer = [False] * len(data["_label_"].unique())
        lcontainer[data[unique_word].idxmax()] = True
        dic_container[unique_word] = lcontainer

    booleandf = pd.DataFrame(dic_container)
    booleandf["_label_"] = data["_label_"]

    return booleandf











