from spacy_transformers.util import get_config, get_model, get_tokenizer
from wasabi import msg
from spacy.util import ensure_path

cache_path = "./trf_models/"

trf_list = [
    'bert-base-uncased',
    'bert-large-uncased',
    'bert-base-cased',
    'bert-large-cased',
    'bert-base-multilingual-uncased',
    'bert-base-multilingual-cased',
    'bert-base-chinese',
    'bert-base-german-cased',
    'bert-large-uncased-whole-word-masking',
    'bert-large-cased-whole-word-masking',
    'bert-large-uncased-whole-word-masking-finetuned-squad',
    'bert-large-cased-whole-word-masking-finetuned-squad',
    'bert-base-cased-finetuned-mrpc',
    'distilbert-base-uncased',
    'distilbert-base-uncased-distilled-squad',
    'gpt2',
    'gpt2-medium',
    'gpt2-large',
    'openai-gpt',
    'roberta-base',
    'roberta-large',
    'roberta-large-mnli',
    'transfo-xl-wt103',
    'xlm-mlm-en-2048',
    'xlm-mlm-ende-1024',
    'xlm-mlm-enfr-1024',
    'xlm-mlm-enro-1024',
    'xlm-mlm-tlm-xnli15-1024',
    'xlm-mlm-xnli15-1024',
    'xlm-clm-enfr-1024',
    'xlm-clm-ende-1024',
    'xlm-mlm-17-1280',
    'xlm-mlm-100-1280',
    'xlnet-base-cased',
    'xlnet-large-cased',
]

def main(
    name: (
        "模型名称", "positional", None, None, trf_list
    ),
    make_cache_dir: (" 创建缓存文件夹", "flag", "mk")
):
    if make_cache_dir:
        c_path = ensure_path(f"{cache_path + name}")
        if c_path.exists():
            msg.warn(f"{cache_path + name} already exists")
        else:
            c_path.mkdir()
            msg.good(f" 缓存文件夹已创建:\t{cache_path}{name}")

    msg.warn("\n================url================\n")

    config = get_config(name)
    config_file = config.pretrained_config_archive_map[name]

    model = get_model(name)
    model_file = model.pretrained_model_archive_map[name]
    msg.text(f"{config_file}\n{model_file}\n")

    vocab = get_tokenizer(name)
    pretrained_vocab_files_map = vocab.pretrained_vocab_files_map
    for vocab_file in pretrained_vocab_files_map.values():
        msg.text(f"{vocab_file[name]}\n")

    msg.warn("\n================url================\n")
    msg.good("\n使用下载工具下载后，将模型文件放入缓存文件夹中。")


if __name__ == "__main__":
    import plac
    plac.call(main)
