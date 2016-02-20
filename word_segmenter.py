# -*- mode: Python; coding: utf-8 -*-
from ChineseWordCorpus import Feature, Feature2, Feature3, ChineseWordCorpus
from crf import CRF, sequence_accuracy
from unittest import TestCase, main

class TestSegmenting(TestCase):

    def setUp(self):
        #get_seed('training')
        self.corpus = ChineseWordCorpus('seed_pku.small', Feature)
        crf = CRF(self.corpus.nodedict, self.corpus.featuredict)
        self.crf =crf

    def test_segmenting(self):
        train = self.corpus[0:7000]
        dev = self.corpus[7000:7050]
        test = self.corpus[8000:10000]
        self.crf.train(train, dev)

        accuracy = sequence_accuracy(self.crf, test)
        print (accuracy)
        self.assertGreaterEqual(accuracy, 0.82)

if __name__ == '__main__':
    main(verbosity=2)


