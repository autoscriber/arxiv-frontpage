import os
import datetime as dt 
from pathlib import Path 

from jinja2 import Template
from radicli import Radicli, Arg

from .utils import console
from .constants import TEMPLATE_PATH, TRAINED_FOLDER, SITE_PATH

cli = Radicli()


@cli.command("download")
def download():
    """Download new data."""
    from .download import main as download_data
    download_data()


@cli.command("index", 
             kind=Arg(help="Can be lunr/simsity"), 
             level=Arg(help="Can be sentence/abstract")
)
def index_cli(kind:str, level:str):
    """Creates index for annotation."""
    from .datastream import DataStream

    DataStream().create_index(level=level, kind=kind)


@cli.command("preprocess")
def preprocess_cli():
    """Dedup and process data for faster processing."""
    from .datastream import DataStream
    DataStream().save_clean_download_stream()


@cli.command("train")
def train():
    """Trains a new model on the data."""
    from .datastream import DataStream
    from .modelling import SentenceModel
    examples = DataStream().get_train_stream()
    SentenceModel().train(examples=examples).to_disk()


@cli.command("pretrain")
def pretrain():
    """Trains a new featurizer, set-fit style."""
    from .datastream import DataStream
    from .modelling import SentenceModel
    examples = DataStream().get_train_stream()
    SentenceModel().pretrain(examples=examples)


@cli.command(
    "build", 
    retrain=Arg("--retrain", "-rt", help="Retrain model?"),
    prep=Arg("--preprocess", "-pr", help="Preprocess again?")
)
def build(retrain: bool = False, prep:bool = False):
    """Build a new site"""
    from .datastream import DataStream
    if prep:
        preprocess_cli()
    if retrain:
        train()
    console.log("Starting site build process")
    sections = DataStream().get_site_content()
    template = Template(Path(TEMPLATE_PATH).read_text())
    rendered = template.render(sections=sections, today=dt.date.today())
    SITE_PATH.write_text(rendered)
    console.log("Site built.")


@cli.command("artifact",
    action=Arg(help="Can be upload/download"),
)
def artifact(action:str):
    """Upload/download from wandb"""
    import wandb
    from dotenv import load_dotenv
    from frontpage.constants import PRETRAINED_FOLDER
    load_dotenv()
    run = wandb.init(os.getenv("WANDB_API_KEY"))
    if action == "upload":
        artifact = wandb.Artifact(name='custom-sbert-emb', type="model")
        artifact.add_dir(local_path=PRETRAINED_FOLDER)
        run = wandb.init(project="arxiv-frontpage", job_type="upload")
        run.log_artifact(artifact)
    if action == "download":
        if not PRETRAINED_FOLDER.exists():
            run = wandb.init(project="arxiv-frontpage", job_type="download")
            artifact = run.use_artifact('custom-sbert-emb:latest')
            console.log(f"Could not find {PRETRAINED_FOLDER}. So will download from wandb.")
            artifact.download(PRETRAINED_FOLDER)
        else:
            console.log(f"{PRETRAINED_FOLDER} already exists. Skip wandb download.")


@cli.command("search")
def search():
    """Annotate new examples."""
    import questionary
    from simsity import load_index
    from .modelling import SentenceModel
    enc = SentenceModel().encoder
    index = load_index("indices/simsity/sentence", encoder=enc)
    while True:
        query = questionary.text("Query:").ask()
        texts, dists = index.query([query], n=5)
        for t in texts:
            print(t)

if __name__ == "__main__":
    cli.run()
