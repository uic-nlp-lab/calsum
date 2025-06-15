#!/usr/bin/env python

"""Create the list of MIMIC-III admission (``hadm_id``) IDs to parse and align.

"""
__author__ = 'Paul Landes'

from typing import Dict, Set, List, Tuple, Any
from dataclasses import dataclass, field
import logging
import random
from collections import OrderedDict
import re
from io import StringIO
from pathlib import Path
import zipfile
import json
from frozendict import frozendict
from zensols.config import Dictable
from zensols.persist import persisted
from zensols.cli import CliHarness, ProgramNameConfigurator
from zensols.mimic import Corpus

logger = logging.getLogger(__name__)
CONFIG = """
[default]
root_dir = ${appenv:root_dir}

[cli]
apps = list: log_cli, app

[log_cli]
class_name = zensols.cli.LogConfigurator
format = ${program:name}: %%(message)s
log_name = ${program:name}
level = info

[selector]
class_name = ${program:name}.AdmissionSelector
corpus = application(zensols.mimic): mimic_corpus
dsprov_file = path: ~/.cache/zensols/dsprov/dsprov-annotations.json
mimicsid_file = path: ~/.cache/zensols/mimicsid/section-id-annotations.zip
output_file = path: ./corpus/mimic-hadms.txt
max_note_count = 50
min_note_count = 2
# samples with replacement
sample_size = 12000

[app]
class_name = ${program:name}.Application
selector = instance: selector

[app_decorator]
option_excludes = set: selector
option_overrides = dict: {'sample_size': {'long_name': 'size'}}
mnemonic_overrides = dict: {'write_ids': 'ids'}
"""


@dataclass
class AdmissionSelector(Dictable):
    _DICTABLE_ATTRIBUTES = {'stats'}
    corpus: Corpus = field(repr=False)
    dsprov_file: Path = field()
    mimicsid_file: Path = field()
    output_file: Path = field()
    max_note_count: int = field()
    min_note_count: int = field()
    sample_size: int = field()

    @property
    @persisted('_counts_by_id')
    def counts_by_id(self) -> Dict[str, int]:
        max_note_count: int = self.max_note_count
        cnts: Tuple[Tuple[int, int], ...] = \
            self.corpus.note_event_persister.get_note_counts()
        return frozendict(map(lambda t: (str(t[0]), t[1]),
                              filter(lambda c: c[1] < max_note_count, cnts)))

    @property
    @persisted('_candidates')
    def candidates(self) -> Set[str]:
        return frozenset(self.counts_by_id.keys())

    @property
    def dsprov_ids(self) -> Set[str]:
        with open(self.dsprov_file) as f:
            return frozenset(set(json.load(f).keys()) & self.candidates)

    @property
    def section_ids(self) -> Set[str]:
        pat: re.Pattern = re.compile(r'.+\/(\d+)-\d+-[a-z]+\.json$')
        with zipfile.ZipFile(self.mimicsid_file, 'r') as zf:
            ids: Set[str] = set(map(lambda m: m.group(1),
                                    filter(lambda m: m is not None,
                                           map(lambda f: pat.match(f),
                                               zf.namelist()))))
        return frozenset(ids & self.candidates)

    def _populate_annotated_ids(self, ids: List[str]):
        dsprov: Set[str] = self.dsprov_ids
        secid: Set[str] = self.section_ids
        inter: Set[str] = dsprov & secid
        ids.extend(dsprov - inter)
        ids.extend(inter)
        ids.extend(secid - inter)

    def _populate_db_ids(self, ids: List[str], sample_size: int):
        sample: List[str] = list(self.candidates - set(ids))
        random.shuffle(sample)
        sample = sample[:self.sample_size]
        ids.extend(sample)
        cnts: Dict[str, int] = self.counts_by_id
        repls: Tuple[str] = tuple(map(
            lambda t: t[0],
            filter(lambda t: t[1] >= self.min_note_count,
                   sorted(map(lambda i: ((i, cnts[i])), ids),
                          key=lambda t: t[1]))))
        ids.clear()
        ids.extend(repls)
        assert len(set(ids)) == len(ids)

    @property
    @persisted('_ids')
    def ids(self) -> Tuple[str, ...]:
        ids: List[str] = []
        self._populate_annotated_ids(ids)
        self._populate_db_ids(ids, self.sample_size)
        return tuple(ids)

    @property
    def stats(self) -> Dict[str, Any]:
        dsprov: Set[str] = self.dsprov_ids
        secid: Set[str] = self.section_ids
        return OrderedDict([
            ('dsprov', len(dsprov)),
            ('secid', len(secid)),
            ('intersection', len(dsprov & secid)),
            ('union', len(dsprov | secid)),
            ('total', len(self.ids))])

    def write_ids(self):
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.write()
        with open(self.output_file, 'w') as f:
            id: str
            for id in self.ids:
                f.write(id)
                f.write('\n')
        logger.info(f'wrote: {self.output_file}')


@dataclass
class Application(object):
    selector: AdmissionSelector = field()

    def stats(self):
        """Print out the ID stats."""
        self.selector.write()

    def write_ids(self):
        """Create clinical note admission selection.

        """
        self.selector.write_ids()


if (__name__ == '__main__'):
    CliHarness(
        src_dir_name='src/bin',
        app_config_resource=StringIO(CONFIG),
        app_config_context=ProgramNameConfigurator(
            None, default='mkids').create_section(),
        proto_args='ids',
        proto_factory_kwargs={'reload_pattern': '^mkids'},
    ).run()
