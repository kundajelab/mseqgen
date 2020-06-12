from mseqgen.generators import MBPNetSequenceGenerator
import argparse
import json
import time
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


parser = argparse.ArgumentParser()

parser.add_argument('--input-json', type=str,
                    help="input directory containing bigWigs and peaks OR "
                    "path to json file containing task information. "
                    "Used as an alternative to input_dir. The keys for each "
                    "task are: 'task_id', 'strand' (0-for plus/unstranded, "
                    "1-for minus), 'signal', 'control' and 'peaks'")
                    

parser.add_argument('--batchgen-params-json', type=str, required=True, 
                    help="path to json file containing batch generation "
                    "params. The required keys are: 'seq_len', 'output_len', "
                    "'max_jitter', 'rev_comp_aug' & 'negative_sampling_rate'. "
                    "The optional keys are 'num_positions' & 'step_size' if "
                    "sampling-mode is 'sequential' and 'num_positions' if "
                    "sampling-mode is 'random'. 'num_positions' refers to the "
                    "number of positions to be sampled from each chromosome. "
                    "In 'sequential' mode if 'num_positions' is -1 then the "
                    "entire chromosome is sampled using the 'step_size'. "
                    "'num_positions' cannot be -1 if mode is 'random'.")
    
parser.add_argument("--bpnet-params-json", 
                    help="path to json file containing any additional "
                    "parameters that are specific to BPNet, for e.g. " 
                    "control_smoothing")

parser.add_argument('--reference-genome', '-g', type=str, required=True, 
                    help="path to the reference genome fasta file")
    
parser.add_argument('--chrom-sizes', '-c', type=str, required=True,
                    help="path to chromosome sizes file")
    
parser.add_argument('--threads', '-t', type=int,
                    help="number of parallel threads for batch generation",
                    default=10)

parser.add_argument('--epochs', '-e', type=int,
                    help="number of training epochs", default=100)

parser.add_argument('--batch-size', '-b', type=int,
                    help="training batch size", default=64)


            
args = parser.parse_args()


# load the input params from json file
with open(args.input_json, "r") as input_json:
    input_params = json.loads(input_json.read())
    
# load the batch gen params from json file
with open(args.batchgen_params_json, "r") as batch_gen_json:
    batch_gen_params = json.loads(batch_gen_json.read())

# load the bpnet params from json file
with open(args.bpnet_params_json, "r") as bpnet_json:
    bpnet_params = json.loads(bpnet_json.read())

                
chroms = ['chr1','chr2','chr3','chr4','chr5','chr6',
          'chr7','chr8','chr9','chr10','chr11','chr12',
          'chr13','chr14','chr15','chr16','chr17','chr18',
          'chr19','chr20','chr21','chr22','chrX','chrY']

val_chroms = ['chr10', 'chr8']
test_chroms = ['chr1']
train_chroms = list(set(chroms).difference(set(val_chroms + test_chroms)))

print(input_params)
print(batch_gen_params)
print(bpnet_params)

seqgen = MBPNetSequenceGenerator(input_params, batch_gen_params, bpnet_params,
                                 args.reference_genome, args.chrom_sizes, 
                                 train_chroms, args.threads, args.epochs, 
                                 args.batch_size)

if seqgen is not None:
    generator = seqgen.gen()

    print(seqgen.get_num_batches_per_epoch())

    t1 = time.time()

    batchCount = 0
    for batchx, batchy in generator:
        print(batchCount)
        batchCount += 1
        if batchCount == seqgen.get_num_batches_per_epoch():
            break

    t2 = time.time()
    print(t2-t1)


"""
python test.py \
--input-json /users/zahoor/mseqgen/tests/input.json \
--batchgen-params-json /users/zahoor/mseqgen/tests/batchgen_params_peaks.json \
--bpnet-params-json /users/zahoor/mseqgen/tests/bpnet_params.json \
--reference-genome /users/zahoor/reference/hg38.genome.fa \
--chrom-sizes /users/zahoor/reference/GRCh38_EBV.chrom.sizes \
--threads 10 \
--epochs 100 \
--batch-size 16 

python test.py \
--input-json /users/zahoor/mseqgen/tests/input2.json \
--batchgen-params-json /users/zahoor/mseqgen/tests/batchgen_params_peaks.json \
--bpnet-params-json /users/zahoor/mseqgen/tests/bpnet_params.json \
--reference-genome /users/zahoor/reference/hg38.genome.fa \
--chrom-sizes /users/zahoor/reference/GRCh38_EBV.chrom.sizes \
--threads 2 \
--epochs 1 \
--batch-size 2 


python test.py \
--input-data /users/zahoor/mseqgen/tests/input-stranded_with_control.json \
--reference-genome /users/zahoor/reference/hg38.genome.fa \
--chrom-sizes /users/zahoor/reference/GRCh38_EBV.chrom.sizes \
--batchgen-params-json /users/zahoor/mseqgen/tests/batchgen_params_peaks.json \
--stranded \
--has-control \
--sampling-mode peaks \
--threads 2 \
--epochs 1 \
--batch-size 2 \
--shuffle \
--bpnet-params-json /users/zahoor/mseqgen/tests/bpnet_params.json


python test.py \
--input-data /users/zahoor/mseqgen/tests/input.json \
--reference-genome /users/zahoor/reference/hg38.genome.fa \
--chrom-sizes /users/zahoor/reference/GRCh38_EBV.chrom.sizes \
--batchgen-params-json /users/zahoor/mseqgen/tests/batchgen_params_peaks.json \
--stranded \
--sampling-mode peaks \
--threads 10 \
--epochs 100 \
--batch-size 16 \
--shuffle \
--bpnet-params-json /users/zahoor/mseqgen/tests/bpnet_params.json

python test.py \
--input-data /users/zahoor/mseqgen/tests/input.json \
--reference-genome /users/zahoor/reference/hg38.genome.fa \
--chrom-sizes /users/zahoor/reference/GRCh38_EBV.chrom.sizes \
--batchgen-params-json /users/zahoor/mseqgen/tests/batchgen_params_sequential.json \
--stranded \
--sampling-mode sequential \
--threads 10 \
--epochs 1 \
--batch-size 16 \
--shuffle \
--bpnet-params-json /users/zahoor/mseqgen/tests/bpnet_params.json

python test.py \
--input-data /users/zahoor/mseqgen/tests/input.json \
--reference-genome /users/zahoor/reference/hg38.genome.fa \
--chrom-sizes /users/zahoor/reference/GRCh38_EBV.chrom.sizes \
--batchgen-params-json /users/zahoor/mseqgen/tests/batchgen_params_random.json \
--stranded \
--sampling-mode random \
--threads 10 \
--epochs 100 \
--batch-size 16 \
--shuffle \
--bpnet-params-json /users/zahoor/mseqgen/tests/bpnet_params.json


"""
