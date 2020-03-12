#!/usr/bin/env python
import plac
import spacy
import GPUtil
import torch
from wasabi import Printer
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy_transformers import TransformersLanguage, TransformersWordPiecer
from spacy_transformers import TransformersTok2Vec
from spacy_transformers.util import get_tokenizer, get_model, get_config
from spacy_transformers.wrapper import TransformersWrapper
from spacy_transformers.model_registry import get_model_function

class TransformersWordPiecerCn(TransformersWordPiecer):
    @classmethod
    def from_pretrained(cls, vocab, trf_name, **cfg):
        tokenizer = get_tokenizer(trf_name)
        tokenizer.pretrained_vocab_files_map["vocab_file"][trf_name] = "./trf_models/bert-base-chinese/vocab.txt"
        model = tokenizer.from_pretrained(trf_name)
        return cls(vocab, model=model, trf_name=trf_name, **cfg)

# is_using_gpu = spacy.prefer_gpu()
# if is_using_gpu:
#     print("Using GPU!")
#     torch.set_default_tensor_type("torch.cuda.FloatTensor")
#     print("GPU Usage")
#     GPUtil.showUtilization()

@plac.annotations(
    path=("Output path", "option", None, str),
    name=("Name of pre-trained model", "option", "n", str),
    lang=("Language code to use", "option", "l", str),
)
def main(path="./spacy_trf_zh", name="bert-base-chinese", lang="zh"):

    msg = Printer()
    msg.info(f"Creating model for '{name}' ({lang})")
    with msg.loading(f"Setting up the pipeline..."):
        nlp = TransformersLanguage(trf_name=name, meta={"lang": lang})
        nlp.add_pipe(nlp.create_pipe("sentencizer"))
        nlp.add_pipe(TransformersWordPiecerCn.from_pretrained(nlp.vocab, name))
        nlp.add_pipe(TransformersTok2Vec.from_pretrained(nlp.vocab, "./trf_models/bert-base-chinese"))
    msg.good("Initialized the model pipeline")
    nlp.to_disk(path)
    msg.good(f"Saved '{name}' ({lang})")
    msg.text(f"Pipeline: {nlp.pipe_names}")
    msg.text(f"Location: {path}")
    with msg.loading("Verifying model loads..."):
        nlp.from_disk(path)
    msg.good("Model loads!")


if __name__ == "__main__":
    plac.call(main)