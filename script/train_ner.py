#!/usr/bin/env python
# coding: utf8
"""Example of training an additional entity type

This script shows how to add a new entity type to an existing pretrained NER
model. To keep the example short and simple, only four sentences are provided
as examples. In practice, you'll need many more — a few hundred would be a
good start. You will also likely need to mix in examples of other entity
types, which might be obtained by running the entity recognizer over unlabelled
sentences, and adding their annotations to the training set.

The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.

After training your model, you can save it to a directory. We recommend
wrapping models as Python packages, for ease of deployment.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.1.0+
Last tested with: v2.1.0
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import *
from spacy.gold import GoldCorpus
from spacy.compat import path2str
from spacy.compat import path2str
from spacy._ml import create_default_optimizer
from timeit import default_timer as timer
import shutil
import srsly
from wasabi import msg
import contextlib

# new entity label
LABEL = ['name', 'organization', 'address', 'position', 'movie', 'scene',
'government', 'company', 'book', 'game']

# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting


@plac.annotations(
    model=("Model name.", "option", "b", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    train_path=("Optional training directory", "option", "t", Path),
    dev_path=("Optional dev directory", "option", "d", Path),
    meta_path=("Optional meta directory", "option", "m", Path),
    use_gpu=("use GPU", "option", "g", int),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(
    model="./zh_vectors_web_ud_lg/model-final",
    new_model_name="zh_vectors_web_ud_clue_lg",
    output_dir="./zh_vectors_web_ud_clue_lg",
    train_path="./clue_spacy_train.jsonl",
    dev_path="./clue_spacy_dev.jsonl",
    meta_path="./meta.json",
    use_gpu=0,
    n_iter=2
):
    import tqdm
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)

    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    for label in LABEL:
        if label not in ner.labels:
            ner.add_label(label)  # add new entity label to entity recognizer

    train_path = ensure_path(train_path)
    dev_path = ensure_path(dev_path)

    if not train_path or not train_path.exists():
        msg.fail("Training data not found", train_path, exits=1)
    if not dev_path or not dev_path.exists():
        msg.fail("Development data not found", dev_path, exits=1)

    if output_dir.exists() and [p for p in output_dir.iterdir() if p.is_dir()]:
        msg.warn(
            "Output directory is not empty",
            "This can lead to unintended side effects when saving the model. "
            "Please use an empty directory or a different path instead. If "
            "the specified output path doesn't exist, the directory will be "
            "created for you.",
        )
    if not output_dir.exists():
        output_dir.mkdir()

    meta = srsly.read_json(meta_path) if meta_path else {}

    # train_path="../clue_spacy_train.jsonl",
    # dev_path="../clue_spacy_dev.json",
    # Prepare training corpus
    msg.text("Counting training words (limit={})".format(0))
    corpus = GoldCorpus(train_path, dev_path, limit=0)
    n_train_words = corpus.count_train()

    
    
    if model is None:   
        optimizer = nlp.begin_training(lambda: corpus.train_tuples, device=use_gpu)
    else:
        optimizer = create_default_optimizer(Model.ops)

    dropout_rates = decaying( 0.2, 0.2, 0.0)
    
    batch_sizes = compounding( 100.0, 1000.0 , 1.001)

    # move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [
        pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]


    # UnboundLocalError: local variable 'has_beam_widths' referenced before assignment
    # fmt: off
    eval_beam_widths=""
    if not eval_beam_widths:
        eval_beam_widths = [1]
    else:
        eval_beam_widths = [int(bw) for bw in eval_beam_widths.split(",")]
        if 1 not in eval_beam_widths:
            eval_beam_widths.append(1)
        eval_beam_widths.sort()
    has_beam_widths = eval_beam_widths != [1]
    row_head, output_stats = _configure_training_output(["ner"], use_gpu, has_beam_widths)
    row_widths = [len(w) for w in row_head]
    row_settings = {"widths": row_widths, "aligns": tuple(["r" for i in row_head]), "spacing": 2}
    # fmt: on
    print("")
    msg.row(row_head, **row_settings)
    msg.row(["-" * width for width in row_settings["widths"]], **row_settings)
    try:
        noise_level = 0.0
        orth_variant_level = 0.0
        gold_preproc = False
        verbose = False

        best_score = 0.0
        with nlp.disable_pipes(*other_pipes):  # only train NER
            for itn in range(n_iter):
                train_docs = corpus.train_docs(
                    nlp,
                    noise_level=noise_level,
                    orth_variant_level=orth_variant_level,
                    gold_preproc=gold_preproc,
                    max_length=0,
                    ignore_misaligned=True,
                )
                words_seen = 0
                with tqdm.tqdm(total=n_train_words, leave=False) as pbar:
                    losses = {}
                    for batch in minibatch_by_words(train_docs, size=batch_sizes):
                        if not batch:
                            continue
                        docs, golds = zip(*batch)
                        nlp.update(
                            docs,
                            golds,
                            sgd=optimizer,
                            drop=next(dropout_rates),
                            losses=losses,
                        )
                        if not int(os.environ.get("LOG_FRIENDLY", 0)):
                            pbar.update(sum(len(doc) for doc in docs))
                        words_seen += sum(len(doc) for doc in docs)
                with nlp.use_params(optimizer.averages):
                    set_env_log(False)
                    epoch_model_path = output_dir / ("model%d" % itn)
                    nlp.to_disk(epoch_model_path)
                    nlp_loaded = load_model_from_path(epoch_model_path)
                    for beam_width in eval_beam_widths:
                        for name, component in nlp_loaded.pipeline:
                            if hasattr(component, "cfg"):
                                component.cfg["beam_width"] = beam_width
                        dev_docs = list(
                            corpus.dev_docs(
                                nlp_loaded,
                                gold_preproc=gold_preproc,
                                ignore_misaligned=True,
                            )
                        )
                        nwords = sum(len(doc_gold[0]) for doc_gold in dev_docs)
                        start_time = timer()
                        scorer = nlp_loaded.evaluate(dev_docs, verbose=verbose)
                        end_time = timer()
                        if use_gpu < 0:
                            gpu_wps = None
                            cpu_wps = nwords / (end_time - start_time)
                        else:
                            gpu_wps = nwords / (end_time - start_time)
                            with Model.use_device("cpu"):
                                nlp_loaded = load_model_from_path(epoch_model_path)
                                for name, component in nlp_loaded.pipeline:
                                    if hasattr(component, "cfg"):
                                        component.cfg["beam_width"] = beam_width
                                dev_docs = list(
                                    corpus.dev_docs(
                                        nlp_loaded,
                                        gold_preproc=gold_preproc,
                                        ignore_misaligned=True,
                                    )
                                )
                                start_time = timer()
                                scorer = nlp_loaded.evaluate(dev_docs, verbose=verbose)
                                end_time = timer()
                                cpu_wps = nwords / (end_time - start_time)
                        acc_loc = output_dir / ("model%d" % itn) / "accuracy.json"
                        srsly.write_json(acc_loc, scorer.scores)

                        # Update model meta.json
                        meta["lang"] = nlp.lang
                        meta["pipeline"] = nlp.pipe_names
                        meta["spacy_version"] = ">=%s" % spacy.__version__
                        if beam_width == 1:
                            meta["speed"] = {
                                "nwords": nwords,
                                "cpu": cpu_wps,
                                "gpu": gpu_wps,
                            }
                            meta["accuracy"] = scorer.scores
                        else:
                            meta.setdefault("beam_accuracy", {})
                            meta.setdefault("beam_speed", {})
                            meta["beam_accuracy"][beam_width] = scorer.scores
                            meta["beam_speed"][beam_width] = {
                                "nwords": nwords,
                                "cpu": cpu_wps,
                                "gpu": gpu_wps,
                            }
                        meta["vectors"] = {
                            "width": nlp.vocab.vectors_length,
                            "vectors": len(nlp.vocab.vectors),
                            "keys": nlp.vocab.vectors.n_keys,
                            "name": nlp.vocab.vectors.name,
                        }
                        meta.setdefault("name", "model%d" % itn)
                        meta.setdefault("version", "0.0.1")
                        meta["labels"] = nlp.meta["labels"]
                        meta_loc = output_dir / ("model%d" % itn) / "meta.json"
                        srsly.write_json(meta_loc, meta)
                        set_env_log(verbose)

                        progress = _get_progress(
                            itn,
                            losses,
                            scorer.scores,
                            output_stats,
                            beam_width=beam_width if has_beam_widths else None,
                            cpu_wps=cpu_wps,
                            gpu_wps=gpu_wps,
                        )

                        msg.row(progress, **row_settings)

    finally:
        with nlp.use_params(optimizer.averages):
            final_model_path = output_dir / "model-final"
            nlp.add_pipe("parser,tagger")
            nlp.to_disk(final_model_path)
        msg.good("Saved model to output directory", final_model_path)
        with msg.loading("Creating best model..."):
            best_model_path = _collate_best_model(meta, output_dir, nlp.pipe_names)
        msg.good("Created best model", best_model_path)


def _score_for_model(meta):
    """ Returns mean score between tasks in pipeline that can be used for early stopping. """
    mean_acc = list()
    pipes = meta["pipeline"]
    acc = meta["accuracy"]
    if "tagger" in pipes:
        mean_acc.append(acc["tags_acc"])
    if "parser" in pipes:
        mean_acc.append((acc["uas"] + acc["las"]) / 2)
    if "ner" in pipes:
        mean_acc.append((acc["ents_p"] + acc["ents_r"] + acc["ents_f"]) / 3)
    if "textcat" in pipes:
        mean_acc.append(acc["textcat_score"])
    return sum(mean_acc) / len(mean_acc)


@contextlib.contextmanager
def _create_progress_bar(total):
    # temp fix to avoid import issues cf https://github.com/explosion/spaCy/issues/4200
    import tqdm

    if int(os.environ.get("LOG_FRIENDLY", 0)):
        yield
    else:
        pbar = tqdm.tqdm(total=total, leave=False)
        yield pbar

def _collate_best_model(meta, output_path, components):
    bests = {}
    for component in components:
        bests[component] = _find_best(output_path, component)
    best_dest = output_path / "model-best"
    shutil.copytree(path2str(output_path / "model-final"), path2str(best_dest))
    for component, best_component_src in bests.items():
        if component == "ner":
            shutil.rmtree(path2str(best_dest / component))
            shutil.copytree(
                path2str(best_component_src / component), path2str(best_dest / component)
            )
            accs = srsly.read_json(best_component_src / "accuracy.json")
            for metric in _get_metrics(component):
                meta["accuracy"][metric] = accs[metric]
    srsly.write_json(best_dest / "meta.json", meta)
    return best_dest


def _find_best(experiment_dir, component):
    accuracies = []
    for epoch_model in experiment_dir.iterdir():
        if epoch_model.is_dir() and epoch_model.parts[-1] != "model-final":
            accs = srsly.read_json(epoch_model / "accuracy.json")
            scores = [accs.get(metric, 0.0) for metric in _get_metrics(component)]
            accuracies.append((scores, epoch_model))
    if accuracies:
        return max(accuracies)[1]
    else:
        return None


def _get_metrics(component):
    if component == "parser":
        return ("las", "uas", "token_acc")
    elif component == "tagger":
        return ("tags_acc",)
    elif component == "ner":
        return ("ents_f", "ents_p", "ents_r")
    return ("token_acc",)


def _configure_training_output(pipeline, use_gpu, has_beam_widths):
    row_head = ["Itn"]
    output_stats = []
    for pipe in pipeline:
        if pipe == "tagger":
            row_head.extend(["Tag Loss ", " Tag %  "])
            output_stats.extend(["tag_loss", "tags_acc"])
        elif pipe == "parser":
            row_head.extend(["Dep Loss ", " UAS  ", " LAS  "])
            output_stats.extend(["dep_loss", "uas", "las"])
        elif pipe == "ner":
            row_head.extend(["NER Loss ", "NER P ", "NER R ", "NER F "])
            output_stats.extend(["ner_loss", "ents_p", "ents_r", "ents_f"])
        elif pipe == "textcat":
            row_head.extend(["Textcat Loss", "Textcat"])
            output_stats.extend(["textcat_loss", "textcat_score"])
    row_head.extend(["Token %", "CPU WPS"])
    output_stats.extend(["token_acc", "cpu_wps"])

    if use_gpu >= 0:
        row_head.extend(["GPU WPS"])
        output_stats.extend(["gpu_wps"])

    if has_beam_widths:
        row_head.insert(1, "Beam W.")
    return row_head, output_stats


def _get_progress(
    itn, losses, dev_scores, output_stats, beam_width=None, cpu_wps=0.0, gpu_wps=0.0
):
    scores = {}
    for stat in output_stats:
        scores[stat] = 0.0
    scores["dep_loss"] = losses.get("parser", 0.0)
    scores["ner_loss"] = losses.get("ner", 0.0)
    scores["tag_loss"] = losses.get("tagger", 0.0)
    scores["textcat_loss"] = losses.get("textcat", 0.0)
    scores["cpu_wps"] = cpu_wps
    scores["gpu_wps"] = gpu_wps or 0.0
    scores.update(dev_scores)
    formatted_scores = []
    for stat in output_stats:
        format_spec = "{:.3f}"
        if stat.endswith("_wps"):
            format_spec = "{:.0f}"
        formatted_scores.append(format_spec.format(scores[stat]))
    result = [itn + 1]
    result.extend(formatted_scores)
    if beam_width is not None:
        result.insert(1, beam_width)
    return result


if __name__ == "__main__":
    plac.call(main)
