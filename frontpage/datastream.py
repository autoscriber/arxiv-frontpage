from typing import Dict, List
import random
import json
import itertools as it
from pathlib import Path

import srsly
from lazylines import LazyLines
from lunr import lunr
from lunr.index import Index

from .constants import DATA_LEVELS, INDICES_FOLDER, LABELS, CONFIG, THRESHOLDS, CLEAN_DOWNLOADS_FOLDER, DOWNLOADS_FOLDER, ANNOT_FOLDER
from .modelling import SentenceModel
from .utils import console, dedup_stream, add_rownum, add_predictions


class DataStream:
    def __init__(self) -> None:
        pass
    
    def get_raw_download_stream(self):
        # Fetch all downloaded files, make sure most recent ones come first
        glob = reversed(list(DOWNLOADS_FOLDER.glob("**/*.jsonl")))
        
        # Make lazy generator for all the items
        stream = it.chain(*list(srsly.read_jsonl(file) for file in glob))
        return stream
    
    def save_clean_download_stream(self):
        stream = dedup_stream(self.get_raw_download_stream(), key="abstract")
        nested = LazyLines(stream).mutate(created=lambda d: d['created'][:10]).nest_by("created")
        for group in nested:
            CLEAN_DOWNLOADS_FOLDER.mkdir(parents=True, exist_ok=True)
            filepath = CLEAN_DOWNLOADS_FOLDER / f"{group['created']}.jsonl"
            g = ({**ex, "created": group['created']} for ex in group['subset'])
            srsly.write_jsonl(filepath, g)
        console.log(f"Cleaned files written in [bold]{CLEAN_DOWNLOADS_FOLDER}[/bold] folder.")
    
    def get_clean_download_stream(self):
        # Fetch all downloaded files, make sure most recent ones come first
        glob = [str(p) for p in CLEAN_DOWNLOADS_FOLDER.glob("**/*.jsonl")]
        arranged_glob = list(reversed(sorted(glob)))
        # Make lazy generator for all the items
        stream = it.chain(*list(srsly.read_jsonl(file) for file in arranged_glob))
        return stream
    
    def get_download_stream(self, level:str="sentence"):
        """Stream of downloaded data, ready for annotation"""
        # Start out with the raw stream
        stream = self.get_clean_download_stream()
        
        # Generate two streams lazily
        abstract_stream = ({"text": ex["abstract"], "sentences": ex["sentences"], "created": ex["created"][:10], "meta": {"created": ex["created"][:10], "url": ex["url"], "title": ex["title"]}} 
                           for ex in stream)
        sentences_stream = ({"text": sent, "meta": {"url": ex["url"]}} 
                            for ex in stream for sent in ex['sentences'])
        stream = abstract_stream if level == "abstract" else sentences_stream
        return stream
    
    def _sentence_data_to_train_format(self, stream):
        """Data ready for training from a sentence-level dataset."""
        for ex in stream:
            # This bit of logic ensures we ignore the `ignore` answer
            outcome = None
            if ex["answer"] == "accept":
                outcome = 1
            if ex["answer"] == "reject":
                outcome = 0
            if outcome is not None:
                yield {
                    "text": ex["text"],
                    ex["label"]: outcome
                }
    
    def _accumulate_train_stream(self, stream) -> List[Dict]:
        """
        This function ensures that we have each `text` appear only
        once and that the categories are nested in the `categories` key.
        """
        return (LazyLines(stream)
                .nest_by("text")
                .mutate(categories=lambda d: {k: v for ex in d['subset'] for k, v in ex.items()})
                .drop("subset")
                .collect())
    
    def get_train_stream(self) -> List[Dict]:
        examples = []
        for label in LABELS:
            path = ANNOT_FOLDER / f"{label}.jsonl"
            for ex in srsly.read_jsonl(path):
                if "categories" not in ex and "cats" in ex:
                    ex["categories"] = ex["cats"]
                examples.append(ex)
        return examples

    def get_lunr_stream(self, query: str, level: str):
        idx_path = self._index_path(kind="lunr", level=level)

        with open(idx_path) as fd:
            reloaded = json.loads(fd.read())
        idx = Index.load(reloaded)
        documents = (LazyLines(self.get_download_stream(level=level))
                     .pipe(add_rownum)
                     .collect())
        return [documents[int(i['ref'])] for i in idx.search(query)]

    def get_ann_stream(self, query: str, level: str):
        from simsity import load_index
        model = SentenceModel()
        idx = load_index(self._index_path(kind="simsity", level=level), encoder=model.encoder)
        texts, scores = idx.query([query], n=150)
        for txt, score in zip(texts, scores):
            example = {"text": txt}
            example["meta"] = {"distance": float(score)}
            yield example
    
    def get_random_stream(self, level:str):
        return (ex for ex in self.get_download_stream(level=level) if random.random() < 0.05)

    def _index_path(self, kind:str, level:str) -> Path:
        """kind is lunr vs. simsity, level is sentence vs. abstract"""
        path = INDICES_FOLDER / kind / level
        if kind == "simsity":
            return path
        path = Path(f"{path}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def create_lunr_index(self, level:str):
        console.log(f"Preparing lunr index for {level}")
        stream = LazyLines(self.get_download_stream(level=level)).pipe(add_rownum).collect()
        index = lunr(ref='idx', fields=('text',), documents=stream)
        serialized = index.serialize()
        with open(self._index_path(kind="lunr", level=level), 'w') as fd:
            json.dump(serialized, fd)
        console.log(f"Lunr index for {level} created")
    
    def create_simsity_index(self, level:str):
        from simsity import create_index
        model = SentenceModel()
        stream = LazyLines(self.get_download_stream(level=level)).map(lambda d: d['text']).collect()
        console.log(f"Preparing simsity index for {level} with {len(stream)} examples.")
        path = self._index_path(kind="simsity", level=level)
        create_index(stream, model.encoder, path=path, batch_size=200, pbar=True)

    def create_index(self, level: str, kind: str):
        if kind == "lunr":
            self.create_lunr_index(level=level)
        if kind == "simsity":
            self.create_simsity_index(level=level)

    def create_indices(self):
        """Index annotation examples for quick annotation."""
        for level in DATA_LEVELS:
            self.create_simsity_index(level=level)
            self.create_lunr_index(level=level)

    def get_site_stream(self):
        model = SentenceModel.from_disk()

        def upper_limit(stream):
            tracker = {lab: 0 for lab in LABELS}
            limit = 50
            for ex in stream:
                for preds in ex['preds']:
                    for name, proba in preds.items():
                        if name in tracker and proba > THRESHOLDS[name] and tracker[name] < limit:
                            tracker[name] += 1
                            if "sections" not in ex:
                                ex['sections'] = []
                            ex['sections'].append(name)
                            ex['sections'] = list(set(ex['sections']))
                            yield ex
                if all(v == limit for v in tracker.values()):
                    break

        console.log("Filtering recent content.")
        return (
            LazyLines(self.get_clean_download_stream())
                .head(1000)
                .pipe(add_predictions, model=model)
                .pipe(upper_limit)
                .collect()
        )
    
    def get_site_content(self):
        site_stream = dedup_stream(self.get_site_stream(), key="abstract")
        sections = {dict(section)['label']: {**dict(section), "content": []} for section in CONFIG.sections}

        def render_html(item, section):
            text = ""
            for sent, pred in zip(item['sentences'], item['preds']):
                proba = pred[section]
                addition = sent
                if proba > THRESHOLDS[section]:
                    proba_val = round(proba, 3)
                    proba_span = f"<span style='font-size: 0.65rem;' class='text-purple-500 font-bold'>{proba_val}</span>"
                    addition = f"<span class='px-1 mx-1 bg-yellow-200'>{addition} {proba_span}</span>"
                text += addition
            return f"<p>{text}</p>"

        for item in site_stream:
            for section in item['sections']:
                editable = item.copy()
                editable['html'] = render_html(editable, section)
                if "categories" not in editable:
                    editable["categories"] = {}
                sections[section]['content'].append(editable)

        for section in sections.keys():
            uniq_content = dedup_stream(sections[section]['content'], key="abstract")
            sections[section]['content'] = reversed(sorted(uniq_content, key=lambda d: d['created']))
        console.log("Sections generated.")
        return list(sections.values())
        
