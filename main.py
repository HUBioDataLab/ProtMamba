from transformers import MambaConfig, MambaForCausalLM, BertTokenizerFast
import torch
from torch.utils.data import DataLoader, random_split
from utils import train, test, UniProtDataset
import wandb    
import argparse
from torchmetrics import Accuracy, Precision, Recall, F1Score

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type = int, default = 100)
parser.add_argument("--batch_size" , type = int, default = 32)
parser.add_argument("--lr", type = float, default = 2e-4)
parser.add_argument("--max_length" , type = int, default = 512)
parser.add_argument("--patience" , type = int, default = 10)
parser.add_argument("--masking_prob", type = float,  default = 0.15)
parser.add_argument("--nhl", help = "number of hidden layers",  default = 32)
parser.add_argument("--weight_decay", type = float, default = 1e-4)
parser.add_argument("--state_size" , type = int, default = 16)
parser.add_argument("--hidden_size" , type = int, default = 768)
parser.add_argument("--random_crop", type=bool, default = True)
parser.add_argument("--device_index", type = int, help = "index of the cuda device", default = 0)
parser.add_argument("--dataset_path", type = str, default = "/media/ubuntu/8TB/mennan/data/uniref100.fasta")
parser.add_argument("--tokenizer_link", type = str, help = "link to huggingface tokenizer", default ="Rostlab/prot_bert" )
parser.add_argument("--model_path", type = str, help = "provide the path to the model state dict if not training from zero", default = None)
parser.add_argument("--wandb_run_id", type = str, help = "provide wandb run id if continuing a previous run", default = "")
parser.add_argument("--wandb_run_name", type = str, help  = "provide wandb run name for new runs")
parser.add_argument("--log_freq", type = int, default = 1000)
args = parser.parse_args()

wandb.login()

config_param = {
        "dataset"          : "Uniref100",
        "architecture"     : "MambaForCausalLM",
        "train_method"     : "masked",
        "epochs"           : args.epochs,
        "batch_size"       : args.batch_size,
        "learning_rate"    : args.lr,
        "max_length"       : args.max_length,
        "patience"         : args.patience,
        "masking_prob"     : args.masking_prob,
        "num_hidden_layers": args.nhl,
        "weight_decay"     : args.weight_decay,
        "hidden_size"      : args.hidden_size,
        "random_crop"      : True,
        "state_size"       : args.state_size
    }
VOCAB_SIZE = 30
device = torch.device("cuda:{}".format(args.device_index) if torch.cuda.is_available() else "cpu")

dataset_path = args.dataset_path

tokenizer_link = args.tokenizer_link
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_link, do_lower_case=False )

batch_size = config_param["batch_size"]
learning_rate = config_param["learning_rate"]

config_mamba = MambaConfig(vocab_size         = tokenizer.vocab_size,
                            eos_token_id      = tokenizer.eos_token_id,
                            bos_token_id      = tokenizer.bos_token_id,
                            pad_token_id      = tokenizer.pad_token_id,
                            hidden_size       = config_param["hidden_size"],
                            num_hidden_layers = config_param["num_hidden_layers"],
                            state_size        = config_param["state_size"]
                            )

dataset = UniProtDataset(   tokenizer    = tokenizer,
                            max_length   = config_param["max_length"],
                            masking_prob = config_param["masking_prob"],
                            random_crop  = config_param["random_crop"],
                            device       = device,
                            dataset_path = dataset_path
                    )

train_size = int(0.8 * len(dataset))
val_size   = int(0.1 * len(dataset))
test_size  = len(dataset) - train_size - val_size
train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size,  val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


accuracy   = Accuracy(task = "multiclass", num_classes = VOCAB_SIZE, average = "macro").to(device)
precision  = Precision(task="multiclass", num_classes= VOCAB_SIZE, average= "macro").to(device)
recall     = Recall(task = "multiclass", num_classes= VOCAB_SIZE, average = "macro").to(device)
f1         = F1Score("multiclass", num_classes = VOCAB_SIZE, average = "macro").to(device)
perplexity = torch.exp
metrics = {"accuracy": accuracy,
           "precision": precision,
           "recall": recall,
           "f1": f1,
           "perplexity": perplexity}
    
model = MambaForCausalLM(config_mamba)
model.to(device)

if args.model_path != None:
    model.load_state_dict(torch.load(args.model_path))

criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr = config_param["learning_rate"], weight_decay = config_param["weight_decay"])

if args.wandb_run_id != None:
    #continue a previous run
    api = wandb.Api()
    run = api.run(f"bilkent_/protmamba/{args.wandb_run_id}")
    run = wandb.init(   
                    resume  = "allow",
                    project = "protmamba",
                    entity  = "bilkent_",
                    id      = args.wandb_run_id,
                    )
else:
    #start a new run
    run = wandb.init(
                    project = "protmamba",
                    config  = config_param,
                    resume  = False,
                    name    = args.wandb_run_name,
                    mode    = "online"
                    )

train(  model            = model,
        tokenizer        = tokenizer,
        train_loader     = train_loader,
        criterion        = criterion,
        val_loader       = val_loader,
        optimizer        = optimizer,
        device           = device,
        epochs           = config_param["epochs"],
        patience         = config_param["patience"],
        mask_token_id    = tokenizer.mask_token_id,
        run              = run,
        metrics          = metrics,
        log_freq         = args.log_freq
        )
test(   model         = model,
        criterion     = criterion,
        test_loader   = test_loader,
        device        = device,
        tokenizer     = tokenizer,
        run           = run,
        metrics       = metrics
        )
run.finish()


