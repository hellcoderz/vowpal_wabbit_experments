# wow! your data can be ANY type you want... does NOT have to be VW examples

import pyvw
from time import time

mapping = {
    "B": 1,
    "I": 2,
    "N": 3
}

rmap = {
    1: "E",
    2: "E",
    3: "N"
}

examples = [
    "watch_N harry_B potter_I",
    "show_N me_N action_B movies_B",
    "jimmy_B kimmel_I show_I",
    "i_N want_N to_N watch_N movies_B on_B demand_I",
    "big_B bang_I theory_I",
    "the_B watch_I",
    "action_B movies_B on_B demand_I",
    "series_B on_B demand_I",
    "what_N is_N the_N score_N of_N flyers_B game_N",
    "watch_N brad_B pitt_I movies_B",
    "taylor_B swift_I",
    "on_I demand_I",
    "avengers_B on_B demand_I",
    "peperoni_B",
    "extra_B",
    "cheese_B",
    "pizza_B",
    "with_N",
    "and_N",
    "scary_B movies_I",
    "scary_B",
    "sunny_B leone_I",
    "miami_B heat_I",
    "what_N is_N the_N san_B fransisco_I score_N",
    "season_B 3_I",
    "episode_B 5_I",
    "game_B of_I throne_I",
    "game_B of_I thrones_I",
    "avengers_B episode_B 6_I season_B 8_I"
]

queries = [
    "watch taylor swift movies",
    "watch taylor swift",
    "watch the watch",
    "i want to watch action movies on demand",
    "watch",
    "watch action movies on demand",
    "watch series on demand",
    "score of flyers game",
    "brad pitt movies",
    "sunny leone series",
    "fylers game score",
    "big bang theory on demand",
    "watch the avengers",
    "i want to order peperoni pizza with extra cheese",
    "show action and comedy movies on demand",
    "peperoni pizza with cheese",
    "show me action movies on hbo",
    "movies",
    "watch x tant",
    "seahawks game score",
    "big bang theory",
    "show me all action movies",
    "watch big bang theory",
    "watch scary movies",
    "what is the miami heat score",
    "get me flyers score",
    "watch big bang theory season 4 episode 2",
    "game of thrones episode 4 season 1",
    "show me avengers movie",
    "free movies",
    "action movies",
    "show me horror comedy movies on hbo",
    "i want peperoni cheese pizza"
]


class SequenceLabeler(pyvw.SearchTask):
    def __init__(self, vw, sch, num_actions):
        # you must must must initialize the parent class
        # this will automatically store self.sch <- sch, self.vw <- vw
        pyvw.SearchTask.__init__(self, vw, sch, num_actions)

        # you can test program options with sch.po_exists
        # and get their values with sch.po_get -> string and
        # sch.po_get_int -> int
        # if sch.po_exists('search'):
        #     print 'found --search'
        #     print '--search value =', sch.po_get('search'), ', type =', type(sch.po_get('search'))

        # set whatever options you want
        sch.set_options(sch.AUTO_HAMMING_LOSS | sch.AUTO_CONDITION_FEATURES)

    def _run(self, sentence):  # it's called _run to remind you that you shouldn't call it directly!
        output = []
        for n in range(len(sentence)):
            pos, word = sentence[n]
            # use "with...as..." to guarantee that the example is finished properly
            with self.vw.example({'w': [word]}) as ex:
                pred = self.sch.predict(examples=ex, my_tag=n + 1, oracle=pos, condition=(n, 'p'))
                output.append(pred)
        return output


class SequenceLabeler2(pyvw.SearchTask):
    def __init__(self, vw, sch, num_actions):
        pyvw.SearchTask.__init__(self, vw, sch, num_actions)

    def _run(self, sentence):
        output = []
        loss = 0.
        for n in range(len(sentence)):
            pos, word = sentence[n]
            prevPred = output[n - 1] if n > 0 else '<s>'
            with self.vw.example({'w': [word], 'p': [prevPred]}) as ex:
                pred = self.sch.predict(examples=ex, my_tag=n + 1, oracle=pos, condition=(n, 'p'))
                output.append(pred)
                if pred != pos:
                    loss += 1.
        self.sch.loss(loss)
        return output


class SequenceLabeler3(pyvw.SearchTask):
    def __init__(self, vw, sch, num_actions):
        pyvw.SearchTask.__init__(self, vw, sch, num_actions)

    def _run(self, sentence):
        output = []
        loss = 0.
        for n in range(len(sentence)):
            pos, word = sentence[n]
            prevPred = output[n - 1] if n > 0 else '<s>'
            prevWord = sentence[n - 1][1] if n > 0 else '<sw>'
            nextWord = sentence[n + 1][1] if n < len(sentence) - 1 else '<nw>'
            with self.vw.example({'w': [word], 'p': [prevPred], 'nw': [nextWord], 'pw': [prevWord]}) as ex:
                pred = self.sch.predict(examples=ex, my_tag=n + 1, oracle=pos, condition=(n, 'p'))
                output.append(pred)
                if pred != pos:
                    loss += 1.
        self.sch.loss(loss)
        return output


class SequenceLabeler4(pyvw.SearchTask):
    def __init__(self, vw, sch, num_actions):
        pyvw.SearchTask.__init__(self, vw, sch, num_actions)

    def _run(self, sentence):
        output = []
        loss = 0.
        for n in range(len(sentence)):
            pos, word = sentence[n]
            prevPred = output[n - 1] if n > 0 else '<s>'
            prevWord = sentence[n - 1][1] if n > 0 else '<sw>'
            nextWord = sentence[n + 1][1] if n < len(sentence) - 1 else '<nw>'
            prevLoc = n - 1 if n > 0 else 100
            nextLoc = n + 1 if n < len(sentence) - 1 else 200
            with self.vw.example({'w': [word], 'p': [prevPred], 'nw': [nextWord], 'pw': [prevWord], 'pl': [prevLoc],
                                  'nl': [nextLoc], 'cl': [n]}) as ex:
                pred = self.sch.predict(examples=ex, my_tag=n + 1, oracle=pos, condition=(n, 'p'))
                output.append(pred)
                if pred != pos:
                    loss += 1.
        self.sch.loss(loss)
        return output


class SequenceLabeler5(pyvw.SearchTask):
    def __init__(self, vw, sch, num_actions):
        pyvw.SearchTask.__init__(self, vw, sch, num_actions)

    def _run(self, sentence):
        output = []
        loss = 0.
        for n in range(len(sentence)):
            pos, word = sentence[n]
            prevPred = output[n - 1] if n > 0 else '<s>'
            prevWord = sentence[n - 1][1] if n > 0 else '<sw>'
            nextWord = sentence[n + 1][1] if n < len(sentence) - 1 else '<nw>'
            with self.vw.example({'w': [word], 'p': [prevPred], 'nw': [nextWord], 'pw': [prevWord], 'cl': [n]}) as ex:
                pred = self.sch.predict(examples=ex, my_tag=n + 1, oracle=pos, condition=(n, 'p'))
                output.append(pred)
                if pred != pos:
                    loss += 1.
        self.sch.loss(loss)
        return output


def load_training_data():
    data = []
    f = open("../../../data/tagger1.train", "rb")
    for line in f:
        line = line.strip()
        if len(line) > 0:
            try:
                data.append(line)
            except:
                print line
    return data


def chunk(lst):
    prev = []
    tag = None
    output = []
    for word, idx in lst:
        if len(prev) == 0:
            prev.append(word)
            tag = idx
        else:
            if (idx == 2 and tag == 1) or (idx == 3 and tag == 3):
                prev.append(word)
            else:
                output.append((" ".join(prev), rmap[tag]))
                prev = [word]
                tag = idx
    if len(prev) > 0:
        output.append((" ".join(prev), rmap[tag]))
    return output

def postprocess(chunked):
    temp = []
    for chunk in chunked:
        temp.append(chunk[0].strip()+"=>"+chunk[1].upper().strip())
    return "|".join(temp)

def preprocess(data):
    dataset = []
    for ex in data:
        parts = ex.strip().split()
        temp = []
        for part in parts:
            word_tag = part.split("_")
            word = word_tag[0].strip().lower()
            tag = word_tag[1].strip().upper()
            temp.append((mapping[tag], word))
        dataset.append(temp)
    return dataset


def test(sequenceLabeler, query):
    try:
        pred = sequenceLabeler.predict([(0, w) for w in query.split()])
        chunked = chunk(zip(query.split(), pred))
        print query, "=>", chunked
        return chunked
    except:
        print "ERROR in prediction"
        return []


def test1(sequenceLabeler):
    # now see the predictions on a test sentence
    print '===============[predicting!]=============='

    start = time()
    for query in queries:
        pred = sequenceLabeler.predict([(0, w) for w in query.split()])
        print ">", query
        print "\t", chunk(zip(query.split(), pred))
        # sequenceLabeler.learn([zip(pred, query.split())])
    end = time()
    print "TIME TAKEN:", (end - start) * 1000, "ms"
