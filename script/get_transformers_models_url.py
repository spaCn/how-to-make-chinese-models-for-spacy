import transformers
from transformers.configuration_auto import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP
from transformers.modeling_auto import ALL_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.tokenization_auto import TOKENIZER_MAPPING
from wasabi import msg
from spacy.util import ensure_path

cache_path = "./trf_models/"
trf_list = [c for c in ALL_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()]


def get_tokenizer(name, local = False):
    """Get a transformers.*Tokenizer class from a name."""
    if not local:
        config = transformers.AutoConfig.from_pretrained(name)

        for config_class, (tokenizer_class_py, tokenizer_class_fast) in TOKENIZER_MAPPING.items():
            if isinstance(config, config_class):
                return tokenizer_class_py

    else:
        name = name.lower()
        if name.startswith("roberta"):
            return transformers.RobertaTokenizer
        elif name.startswith("distilbert"):
            return transformers.DistilBertTokenizer
        elif name.startswith("bert"):
            return transformers.BertTokenizer
        elif name.startswith("xlnet"):
            return transformers.XLNetTokenizer
        elif name.startswith("gpt2"):
            return transformers.GPT2Tokenizer
        elif name.startswith("xlm"):
            return transformers.XLMTokenizer
        else:
            raise ValueError(f"本地Class中未找到 '{name}'的配置，请去掉-local试一下。")

def main(
    name: (
        "模型名称", "positional", None, None, trf_list
    ),
    make_cache_dir: (" 创建缓存文件夹", "flag", "mk"),
    use_local_class: ("不使用网络读取", "flag", "local")
):
    if make_cache_dir:
        c_path = ensure_path(f"{cache_path + name}")
        if c_path.exists():
            msg.warn(f"{cache_path + name} already exists")
        else:
            c_path.mkdir()
            msg.good(f" 缓存文件夹已创建:\t{cache_path}{name}")

    msg.warn("\n================url================\n")

    config_file = ALL_PRETRAINED_CONFIG_ARCHIVE_MAP[name]

    model_file = ALL_PRETRAINED_MODEL_ARCHIVE_MAP[name]
    msg.text(f"{config_file}\n{model_file}\n")

    vocab = get_tokenizer(name, use_local_class)
    pretrained_vocab_files_map = vocab.pretrained_vocab_files_map
    for vocab_file in pretrained_vocab_files_map.values():
        msg.text(f"{vocab_file[name]}\n")

    msg.warn("\n================url================\n")
    msg.good("\n使用下载工具下载后，将模型文件放入缓存文件夹中。")


if __name__ == "__main__":
    import plac
    plac.call(main)
