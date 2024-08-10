from string import punctuation
from nltk.stem import WordNetLemmatizer
from collections import Counter


stop_words = [
    "a", "an", "i", "you", "u", "to", "for", "and", "the", "have", "had", "is", "are", "im", "they", "we", "am"]
lemmatizer = WordNetLemmatizer()
punc = punctuation


def pre_process_text(container):
    container = [[s.lower() for s in sentence.split()] for sentence in container]
    container = [[s.strip() for s in sentence] for sentence in container]

    for ind, sentence in enumerate(container):
        for p in punc:
            container[ind] = [word.replace(p, "") for word in sentence]

    return [[' '.join(sentence)] for sentence in container]


def pre_process_sentence(sentence):
    handler = [w.lower() for w in sentence.split() if w not in punc]

    for idx, word in enumerate(handler):
        for p in punc:
            while p in handler[idx]:
                handler[idx] = word.replace(p,"")

    return ' '.join(handler)


def lemmatize_data(container):
    container = [word for sentence in container for word in sentence]

    handler = [lemmatizer.lemmatize(word, pos='v') for sentence in container for word in sentence.split() if word not in stop_words and len(word) > 1]

    if not handler:
        raise ValueError("No words to lemmatize :|")

    return handler


def lemmatize_text(text):
    return ' '.join(lemmatizer.lemmatize(w, pos='v') for w in text.split() if w not in stop_words and len(w) > 1)


def lemmatize_text_column(text_column):
    return [lemmatize_text(text) for text in text_column]

""" 
unnecessary atm 👇
"""
# def library_counting(cleaned_data):
#     return Counter(' '.join(cleaned_data).split())
#
#
# def manual_counting(cleaned_data):
#     dic = dict()
#     for word in cleaned_data:
#         dic[word] = dic.get(word, 0) + 1
#     return dic
