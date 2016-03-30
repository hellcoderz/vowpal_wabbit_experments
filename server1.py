import sys
from utils import examples, rmap, mapping, queries, SequenceLabeler4, SequenceLabeler3, chunk, preprocess, test, postprocess
import pyvw
from time import time

samples = [
    ("watch harry potter", "watch=>N|harry potter=>E"),
    ("harry potter", "harry potter=>E"),
    ("watch big bang", "watch=>N|big bang=>E"),
    ("big bang", "big bang=>E"),
    ("show me harry potter", "show me=>N|harry potter=>E"),
    ("show me harry potter movies", "show me=>N|harry potter=>E|movies=>E"),
    ("action movies", "action=>E|movies=>E"),
    ("watch action movies", "watch=>N|action=>E|movies=>E")
]


def train(sequenceLabeler, data):
    sequenceLabeler.learn(preprocess([data]))


def testit(sequenceLabeler):
    passed = 0
    for sample in samples:
        pred = postprocess(test(sequenceLabeler, sample[0]))
        if pred == sample[1]:
            passed += 1

    print "\n======== ACCURACY:[", (passed*1.0/len(samples))*100, "% ] ======"
    print "====================================\n"

if __name__ == "__main__":
    fname = "tagger2.bin"
    vw = pyvw.vw("--search 3 --quiet --search_task hook --ring_size 2048 -f " + fname)
    sequenceLabeler = vw.init_search_task(SequenceLabeler3)

    train(sequenceLabeler, "watch_N big_B bang_I")
    testit(sequenceLabeler)

    train(sequenceLabeler, "harry_B potter_I")
    testit(sequenceLabeler)

    train(sequenceLabeler, "show_N me_N action_B movies_B")
    testit(sequenceLabeler)
    vw.finish()
