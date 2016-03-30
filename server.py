import sys
from data import examples, rmap, mapping, queries

sys.path.insert(0, '../')
import pyvw
from time import time


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


def train(dataset, fname, N=20, load=False):
    # initialize VW as usual, but use 'hook' as the search_task
    vw = None
    if load:
        vw = pyvw.vw("--quiet -i " + fname + " -f " + fname)
    else:
        vw = pyvw.vw("--search 3 --quiet --search_task hook --ring_size 2048 -f " + fname)
    # tell VW to construct your search task object
    # sequenceLabeler = vw.init_search_task(SequenceLabeler)
    # sequenceLabeler = vw.init_search_task(SequenceLabeler2)
    # sequenceLabeler = vw.init_search_task(SequenceLabeler3)
    sequenceLabeler = vw.init_search_task(SequenceLabeler4)
    # train it on the above dataset ten times; the my_dataset.__iter__ feeds into _run above
    start = time()
    print '===============[training!]================'
    for i in xrange(N):
        sequenceLabeler.learn(dataset)
    end = time()
    print "TIME TAKEN:", (end - start) * 1000, "ms"
    return vw, sequenceLabeler


def load_data():
    # my_dataset = preprocess(examples)
    dataset = preprocess(load_training_data())
    print "dataset size:", len(dataset)
    return dataset


def load(fname):
    vw = pyvw.vw("--quiet -i " + fname)
    # tell VW to construct your search task object
    # sequenceLabeler = vw.init_search_task(SequenceLabeler)
    # sequenceLabeler = vw.init_search_task(SequenceLabeler2)
    # sequenceLabeler = vw.init_search_task(SequenceLabeler3)
    sequenceLabeler = vw.init_search_task(SequenceLabeler4)
    return vw, sequenceLabeler


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


if __name__ == "__main__":
    fname = "tagger.bin"

    data = []
    data = preprocess(examples) + data
    # vw, sequenceLabeler = train(data, fname, 20)
    vw, sequenceLabeler = load(fname)
    # vw, sequenceLabeler = train(preprocess(examples), fname, 50, True)
    test1(sequenceLabeler)
    vw.finish()
