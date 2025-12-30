import datetime as dt 
from pathlib import Path
from typing import List

import srsly
import tqdm
import arxiv
from arxiv import Result
from retry import retry 
import spacy
from spacy.language import Language
from .types import ArxivArticle
from rich.console import Console 

console = Console()

from .constants import DOWNLOADS_FOLDER

def age_in_days(res: Result) -> float:
    """Get total seconds from now from Arxiv result"""
    now = dt.datetime.now(dt.timezone.utc)
    return (now - res.published).total_seconds() / 3600 / 24


def parse(res: Result, nlp: Language) -> ArxivArticle:
    """Parse proper Pydantic object from Arxiv"""
    summary = res.summary.replace("\n", " ")
    doc = nlp(summary)
    sents = [s.text for s in doc.sents]
    
    return ArxivArticle(
        created=str(res.published)[:19], 
        title=str(res.title),
        abstract=summary,
        sentences=sents,
        url=str(res.entry_id)
    )

@retry(tries=5, delay=1, backoff=2)
def main():
    # arXiv now redirects API traffic to HTTPS, which the client class does not
    # follow. Override the endpoint to avoid HTTP 301 errors.
    arxiv.Client.query_url_format = "https://export.arxiv.org/api/query?{}"

    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])
    console.log(f"Starting arxiv search.")
    # Use the common token "and" because it appears in most papers; combined
    # with SubmittedDate sorting this fetches the latest ~200 papers, which we
    # then filter locally to recent CS items.
    items = arxiv.Search(
        query="and",
        max_results=200,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    results = list(items.results())

    console.log(f"Found {len(results)} results.")

    articles = [dict(parse(r, nlp=nlp)) 
                for r in tqdm.tqdm(results) 
                if age_in_days(r) < 2.5 and r.primary_category.startswith("cs")]

    # Calculate the age of the articles in days just for logging purposes
    dist = [age_in_days(r) for r in results]
    if dist:
        console.log(f"Minimum article age: {min(dist)}")
        console.log(f"Maximum article age: {max(dist)}")

    # Convert the articles to a dictionary for faster lookup
    articles_dict = {article["title"]: article for article in articles}
    most_recent = list(sorted(DOWNLOADS_FOLDER.glob("*.jsonl")))[-1]
    old_articles_dict = {article["title"]: article for article in srsly.read_jsonl(most_recent)}

    # Find the new and old articles
    new_articles = [article for title, article in articles_dict.items() if title not in old_articles_dict.keys()]
    # Find the old articles
    old_articles = [article for title, article in articles_dict.items() if title in old_articles_dict.keys()]

    # Log the number of old and new articles
    if old_articles:
        console.log(f"Found {len(old_articles)} old articles in current batch. Skipping.")
        
    # If there are new articles, write them to the downloads folder
    if new_articles:
        console.log(f"Found {len(new_articles)} new articles in current batch to write.")
        filename = str(dt.datetime.now()).replace(" ", "-")[:13] + "h.jsonl" # Format the filename
        srsly.write_jsonl(DOWNLOADS_FOLDER / filename, new_articles) # Write the new articles to the downloads folder
        console.log(f"Wrote {len(new_articles)} articles into {filename}.") # Log the number of new articles written
