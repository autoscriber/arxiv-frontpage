import os
from pathlib import Path
from typing import Dict, List

import srsly
import vertexai
from vertexai.generative_models import GenerativeModel

from .constants import CONFIG, LABELS, ANNOT_FOLDER
from .datastream import DataStream
from .utils import console


def _choose_label(text: str, model: GenerativeModel, labels: List[str]) -> str:
    prompt = (
        "You are labeling arXiv content with one label from this list: "
        f"{labels}. Return only the label.\n\nText:\n{text}"
    )
    resp = model.generate_content(prompt)
    # vertexai response provides .text convenience
    return (resp.text or "").strip()


def _to_categories(label: str, labels: List[str]) -> Dict[str, int]:
    categories = {lab: 0 for lab in labels}
    if label in categories:
        categories[label] = 1
    return categories


def run(model: str = "gemini-2.5-flash", level: str = "abstract", limit: int = 300):
    """Generate annotations with Gemini and save in ANNOT_FOLDER."""
    # Prefer environment values; if unset, default to project root creds and project id.
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/antonellaschiavoni/Projects/arxiv-frontpage/google_cred.json"
    if "GCP_PROJECT" not in os.environ:
        os.environ["GCP_PROJECT"] = "gemini-423213"

    creds_path = Path(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "google_cred.json"))
    if not creds_path.exists():
        raise FileNotFoundError(
            f"Missing credentials file at {creds_path}. "
            "Set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON."
        )
    project = os.getenv("GCP_PROJECT") or CONFIG.project_id if hasattr(CONFIG, "project_id") else None
    location = os.getenv("GCP_LOCATION", "us-central1")
    if not project:
        raise ValueError("Set GCP_PROJECT env or add project_id to config.")

    vertexai.init(project=project, location=location)
    gen_model = GenerativeModel(model)
    ds = DataStream()
    stream = ds.get_download_stream(level=level)

    examples = []
    for ex in stream:
        text = ex.get("text") or ex.get("abstract") or " ".join(ex.get("sentences", []))
        if not text:
            continue
        label = _choose_label(text, model=gen_model, labels=LABELS)
        categories = _to_categories(label, LABELS)
        examples.append({"text": text, "categories": categories})
        if len(examples) >= limit:
            break

    ANNOT_FOLDER.mkdir(parents=True, exist_ok=True)
    for label in LABELS:
        subset = [ex for ex in examples if ex["categories"].get(label) == 1]
        path = ANNOT_FOLDER / f"{label}.jsonl"
        srsly.write_jsonl(path, subset)
        console.log(f"Wrote {len(subset)} examples to {path}")


if __name__ == "__main__":
    run()

