from transformers import MambaForCausalLM, AutoTokenizer
import sys
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--device_index", type = int, help = "index of the cuda device", default = 0)
parser.add_argument("--tokenizer_path", type = str, help = "path to the pretrained tokenizer, either huggingface hub directory or local directory")
parser.add_argument("--model_load_path", type = str, help = "the path to the model state dict if not training from zero" )
parser.add_argument("--dataset_path", type = str, help = "the path to local directory or huggingface hub directory for the DNMT dataset", default = "Mennan/dnmt")
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


def create_gen_file(
                   num_seq : int = 1000,
                   max_length : int = 512,
                   min_length : int = 200,
                   save_file_path : str):
    partitions = [0.1 * i for i in range(1, 10)]
    device = torch.device(f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, 
                                            padding_side='left')

    model = MambaForCausalLM.from_pretrained(args.model_load_path)
    model.to(device)
    model.eval()

    test_dataset = load_dataset("Mennan/dnmt")['test']
    test_dataset = test_dataset.remove_columns( ["cluster_id", "protein_id"])
    test_dataset = test_dataset.with_format(type = "torch")
    with open(save_file_path, 'w') as file:
        for i in range(len(test_dataset)):
            input_seq = test_dataset[i]
            seq_len = len(input_seq['sequence'])
            input_list = [input_seq['sequence'][:int(seq_len * i)] for i in partitions]
            encoded_input_list = tokenizer(input_list, padding = "max_length", max_length = max_len, truncation = True, return_tensors = 'pt').to(device)
            input_ids = encoded_input_list['input_ids']
            gen_seq = model.generate(input_ids, 
                    max_length = max_length,
                    min_length = min_length,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature = args.temperature,
                top_k = args.top_k,
                top_p = args.top_p,
                no_repeat_ngram_size = args.no_repeat_ngram_size,
                repetition_penalty = args.repetition_penalty)
            decoded_gen_seqs = tokenizer.batch_decode(gen_seq, skip_special_tokens = True)
            file.write(f"{input_seq['sequence']}")
            file.flush()
            for seq in decoded_gen_seqs:
                file.write(f"{seq.replace(' ', '')}\n")
            file.write("\n")
            file.flush()

if __name__ == "__main__":
    try:
        create_gen_file(num_seq = args.num_sequences_to_generate,
                        max_length = args.max_length,
                        min_length = args.min_length,
                        save_file_path = args.save_file_path)
    except Exception as e:
        print(f"Program failed: {str(e)}")
        sys.exit(1)