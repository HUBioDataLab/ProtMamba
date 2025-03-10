from transformers import MambaForCausalLM, AutoTokenizer
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device_index", type = int, help = "index of the cuda device", default = 0)
parser.add_argument("--tokenizer_path", type = str, help = "path to the pretrained tokenizer, either huggingface hub directory or local directory")
parser.add_argument("--model_load_path", type = str, help = "the path to the model state dict if not training from zero" )
parser.add_argument("--save_file_path", type = str, help = "the path to save the generated sequences" )
parser.add_argument("--num_sequences_to_generate", type = int, default = 1280)
parser.add_argument("--min_length", type = int, default = 200)
parser.add_argument("--max_length", type = int, default = 512)
parser.add_argument("--temperature", type = float, default = 0.8)
parser.add_argument("--top_k", type = int, default = 5)
parser.add_argument("--top_p", type = float, default = .9)
parser.add_argument("--no_repeat_ngram_size", type = int, default = 3)
parser.add_argument("--repetition_penalty", type = float, default = 1.2)
parser.add_argument("--batch_size", type = int, default = 64)
args = parser.parse_args()

model = MambaForCausalLM.from_pretrained(args.model_load_path)
device = torch.device(f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu") 
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,padding_side='left')

input_ids = tokenizer.encode(tokenizer.cls_token, return_tensors="pt").to(device)

file = open(f"{args.save_file_path}.txt", 'w')
for i in range(args.num_sequences_to_generate / args.batch_size):    
    gen_seq = model.generate( input_ids,
                        max_length = args.max_length,
                        min_length = args.min_length,
                        num_return_sequences = args.batch_size,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature = args.temperature,
                    top_k = args.top_k,
                    top_p = args.top_p,
                    no_repeat_ngram_size = args.no_repeat_ngram_size,
                    repetition_penalty = args.repetition_penalty)
    generated_sequences = tokenizer.batch_decode(gen_seq, skip_special_tokens = True)
    for seq in generated_sequences:
        file.write(f"{seq.replace(' ', '')}\n")
        file.flush()
    print(f"Generated and saved { (i + 1 ) * args.batch_size} sequences.")
