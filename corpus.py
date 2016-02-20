# -*- mode: Python; coding: utf-8 -*-


from abc import ABCMeta, abstractmethod
from csv import reader as csv_reader
from glob import glob
import json
from os.path import basename, dirname, split, splitext


class Document(object):
    """This is a superclass for all the features"""

    max_display_data = 10 # limit for data abbreviation

    def __init__(self, data, node=None, source=None):
        self.data = data
        self.node = node
        self.source = source
        self.fv = []

    def __repr__(self):
        return ("<%s: %s>" % (self.label, self.abbrev()) if self.label else
                "%s" % self.abbrev())

    def abbrev(self):
        return (self.data if len(self.data) < self.max_display_data else
                self.data[0:self.max_display_data] + "...")

    def features(self):
        """A list of features that characterize this document."""
        return [self.data]


