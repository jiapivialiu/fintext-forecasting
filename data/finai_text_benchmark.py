# load financial text benchmark datasets
from datasets import load_dataset

bigdata_train = load_dataset("TheFinAI/flare-sm-bigdata", split="train")
bigdata_valid = load_dataset("TheFinAI/flare-sm-bigdata", split="validation")
bigdata_test = load_dataset("TheFinAI/flare-sm-bigdata", split="test")

acl_train = load_dataset("TheFinAI/flare-sm-acl", split="train")
acl_valid = load_dataset("TheFinAI/flare-sm-acl", split="valid")
acl_test = load_dataset("TheFinAI/flare-sm-acl", split="test")

cikm_train = load_dataset("TheFinAI/flare-sm-cikm", split="train")
cikm_valid = load_dataset("TheFinAI/flare-sm-cikm", split="valid")
cikm_test = load_dataset("TheFinAI/flare-sm-cikm", split="test")

bigdata_train_df = bigdata_train.to_pandas()[['gold', 'text']] # 0: rise, 1: fall
bigdata_valid_df = bigdata_valid.to_pandas()[['gold', 'text']]
bigdata_test_df = bigdata_test.to_pandas()[['gold', 'text']]

acl_train_df = acl_train.to_pandas()[['gold', 'text']]
acl_valid_df = acl_valid.to_pandas()[['gold', 'text']]
acl_test_df = acl_test.to_pandas()[['gold', 'text']]

cikm_train_df = cikm_train.to_pandas()[['gold', 'text']]
cikm_valid_df = cikm_valid.to_pandas()[['gold', 'text']]
cikm_test_df = cikm_test.to_pandas()[['gold', 'text']]
