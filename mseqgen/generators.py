"""
    This module contains classes for all the sequence data generators

    Classes:
    
        MSequenceGenerator - The main base class for all generators. 
            Multi task batch data generation for training deep neural
            networks on high-throughput sequencing data of various
            geonmics assays
    
        MBPNetSequenceGenerator - Derives from MSequenceGenerator.
            Multi task batch data generation for training BPNet
            on high-throughput sequencing data of various
            geonmics assays
            
            
    License:
    
    MIT License

    Copyright (c) 2020 Kundaje Lab

    Permission is hereby granted, free of charge, to any person 
    obtaining a copy of this software and associated documentation
    files (the "Software"), to deal in the Software without 
    restriction, including without limitation the rights to use, copy,
    modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be 
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
    BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
    ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

"""

import json
import logging
import multiprocessing as mp
import numpy as np
import os
import pandas as pd 
import pyBigWig
import pyfaidx
import random

from mseqgen import sequtils
from mseqgen import quietexception
from queue import Queue
from scipy.ndimage import gaussian_filter1d
from threading import Thread

class MSequenceGenerator:
    
    """ Multi task batch data generation for training deep neural
        networks on high-throughput sequencing data of various
        geonmics assays
    
        Args:
            input_params (dict): python dictionary with information 
                about the input data. Contains the following keys -
                
                "data" (str)
                ------------
                the path to the data directory containing the signal &
                peaks files for each task OR path to json file 
                containing task information. See README for more
                information on how the data directory has to be 
                organized and the format of the json file if you 
                prefer to use a json file
                
                "stranded" (boolean)
                --------------------
                True if data is stranded
                
                "has_control" (boolean)
                -----------------------
                True if control data has been included 
                
            batch_gen_params (dictionary): python dictionaru with batch
                generation parameters. Contains the following keys - 
            
                "input_seq_len" (int)
                ---------------------
                length of input DNA sequence
                
                "output_len" (int)
                ------------------
                length of output profile
                
                "max_jitter" (int)
                ------------------
                maximum value for randomized jitter to offset the peaks
                from the exact center of the input
                
                "rev_comp_aug" (boolean)
                ------------------------
                enable reverse complement augmentation
                
                "negative_sampling_rate" (float)
                --------------------------------
                the fraction of batch_size that determines how many 
                negative samples are added to each batch
            
                "sampling_mode" (str)
                ---------------------
                the mode of sampling chromosome positions - one of
                ['peaks', 'sequential', 'random']
                
                shuffle (boolean)
                -----------------
                specify whether input data is shuffled at the 
                begininning of each epoch
                
                "mode" (str)
                ------------
                "train", "val" or "test"
                 
                "num_positions" (int)
                ---------------------
                specify how many chromosome positions to sample if 
                sampling_mode is "sequential" or "random". Can be 
                omitted if sampling_mode is "peaks", has no effect if
                present.
                 
                "step_size" (int)
                -----------------
                specify step size for sampling chromosome positions if 
                sampling_mode is "sequential". Can be omitted if 
                sampling_mode is "peaks" or "random", has no effect if
                present.

            reference_genome (str): the path to the reference genome 
                fasta file
                
            chrom_sizes (str): path to the chromosome sizes file
            
            chroms (str): the list of chromosomes that will be sampled
                for batch generation
                
            num_threads (int): number of parallel threads for batch
                generation
                
            epochs (int): number of iterations for looping over input
                data
                
            batch_size (int): size of each generated batch of data

            
        Attributes:
            sampling_mode (str): the mode of sampling chromosome 
                positions; one of  ['peaks', 'sequential', 'random'].
            
            mode (str): "train", "val" or "test"

            tasks (collections.OrderedDict): dictionary of input tasks
                derived either from input_data dir or taken from 
                input_data json
            
            num_tasks (int): the number of tasks in 'tasks'
            
            reference (str): the path to the reference genome 
                fasta file
            
            chroms (list): the list of chromosomes that will be sampled
                for batch generation
            
            chrom_sizes_df (pandas.Dataframe): dataframe of the 
                chromosomes and their corresponding sizes
            
            num_threads (int): number of parallel threads for batch
                generation
            
            epochs (int): number of iterations for looping over input
                data
            
            batch_size (int): size of each generated batch of data
            shuffle (boolean):  whether input data is shuffled at the
            begininning of each epoch
            
            input_flank (int): one half of input sequence length
            
            output_flank (int): one half of output sequence length
            
            max_jitter (int): the maximum absolute value of jitter to
                vary the position of the peak summit to left or right
                of the exact center of the input sequence. Range is
                -max_jitter to +max_jitter.
            
            negative_sampling_rate (float): Use a positive value > 0.0
                to specify how many negative samples will be added to
                each batch. num_negative_samples = 
                negative_sampling_rate * batch_size. Ignored if 
                --sampling_mode is not 'peaks', and --mode is not 
                'train'
            
            rev_comp_aug (boolean): specify whether reverse complement
                augmentation should be applied to each batch of data.
                If True, the size of the generated batch is doubled 
                (i.e batch_size*2 or if negative samples are added then
                (batch_size + num_negative_samples)*2). Ignored if 
                --mode is not 'train'
            
            ready_for_next_epoch (boolean): flag to control batch 
                generation for the next epoch. The consumer of the 
                generator is required to send this signal using 
                'set_ready_for_next_epoch'. This protocol is required
                so that excessive and unnecessary batches are not 
                generated if they will not be consumed 
            
            stop (boolean): flag to indicate that batch generation 
                should be terminated after the current epoch
    """
    

    def __init__(self, input_params, batch_gen_params, reference_genome, 
                 chrom_sizes, chroms, num_threads, epochs, batch_size):
        
        # sampling mode to get chromosome positions
        self.sampling_mode = batch_gen_params['sampling_mode']

        # ML task mode "train", "val" or "test"
        self.mode = batch_gen_params['mode']
        
        # check if at least one of the two input modes is present
        if not os.path.isdir(input_params['data']) and \
            os.path.splitext(input_params['data'])[1] != '.json':
            raise quietexception.QuietException(
                "Either input directory or input json must be specified. "
                "None found.")
        
        # load the input tasks either from the input dir or from
        # the input json
        if os.path.isdir(input_params['data']):
            self.tasks = sequtils.getInputTasks(
                input_params['data'], stranded=input_params['stranded'], 
                has_control=input_params['has_control'],
                require_peaks=(self.sampling_mode == 'peaks'), 
                mode=self.mode)
        else:
            # make sure the input_data json file exists
            if not os.path.isfile(input_params['data']):
                raise quietexception.QuietException(
                    "File not found: {}", input_params['data'])
        
            with open(input_params['data'], 'r') as inp_json:
                self.tasks = json.loads(inp_json.read())

        # check if the reference genome file exists
        if not os.path.isfile(reference_genome):
            raise quietexception.QuietException(
                "File not found: {}", reference_genome)
        
        # check if the chrom_sizes file exists
        if not os.path.isfile(chrom_sizes):
            raise quietexception.QuietException(
                "File not found: {}", chrom_sizes)

        self.num_tasks = len(list(self.tasks.keys()))
        self.reference = reference_genome

        # read the chrom sizes into a dataframe 
        self.chrom_sizes_df = pd.read_csv(chrom_sizes, sep = '\t', 
                              header=None, names = ['chrom', 'size']) 

        # chromosome list
        self.chroms = chroms
        
        # keep only those chrom_sizes rows corresponding to the 
        # required chromosomes
        self.chrom_sizes_df = self.chrom_sizes_df[
            self.chrom_sizes_df['chrom'].isin(self.chroms)]

        # generate a new column for sampling weights of the chromosomes
        self.chrom_sizes_df['weights'] = (self.chrom_sizes_df['size'] / 
                                          self.chrom_sizes_df['size'].sum())

        self.num_threads = num_threads
        self.epochs = epochs
        self.batch_size = batch_size

        # rest of batch generation parameters
        self.input_flank =  batch_gen_params['input_seq_len'] // 2
        self.output_flank = batch_gen_params['output_len'] // 2        
        self.max_jitter = batch_gen_params['max_jitter']
        self.negative_sampling_rate = batch_gen_params['negative_sampling_rate']
        self.rev_comp_aug = batch_gen_params['rev_comp_aug']
        self.shuffle = batch_gen_params['shuffle']
        
        # control batch generation for next epoch
        # if the value is not set to True, batches are not generated
        # Use an external controller to set value to True/False
        self.ready_for_next_epoch = False
        
        # (early) stopping flag
        self.stop = False
        
        if self.sampling_mode == 'peaks':
            
            # get a pandas dataframe for the peak positions
            # Note - we need the 'tasks' dictionary so we can access
            # the peaks.bed files from the paths available in the 
            # dictionary
            self.data = sequtils.getPeakPositions(
                self.tasks, self.chroms, 
                self.chrom_sizes_df[['chrom', 'size']], self.input_flank)  
            
        elif self.sampling_mode == 'sequential':
            
            if 'num_positions' not in batch_gen_params:
                raise quietexception.QuietException(
                    "Key not found in batch_gen_params_json: 'num_positions'. " 
                    "Required for sequential sampling mode")

            if 'step_size' not in batch_gen_params:
                raise quietexception.QuietException(
                    "Key not found in batch_gen_params_json: 'step_size'. " 
                    "Required for sequential sampling mode")

            # get a pandas dataframe with sequential positions at 
            # regular intervals
            self.data = sequtils.getChromPositions(
                self.chroms, self.chrom_sizes_df[['chrom', 'size']], 
                self.input_flank, mode=self.sampling_mode, 
                num_positions=batch_gen_params['num_positions'], 
                step=batch_gen_params['step_size'])

            self.max_jitter = 0

        elif self.sampling_mode == 'random':
            
            if 'num_positions' not in batch_gen_params:
                raise quietexception.QuietException(
                    "Key not found in batch_gen_params_json: 'num_positions'. "
                    "Required for random sampling mode")
            
            # get a pandas dataframe with random positions
            self.data = sequtils.getChromPositions(
                self.chroms, self.chrom_sizes_df[['chrom', 'size']], 
                self.input_flank, mode=self.sampling_mode, 
                num_positions=batch_gen_params['num_positions'])

            self.max_jitter = 0
    
    
    def get_num_batches_per_epoch(self):
        """
            The number of batches per epoch
            
            Returns:
                int: number of batches of data generated in each epoch
        """
        
        return self.data.shape[0] // self.batch_size
        

    def set_ready_for_next_epoch(self):
        """ Set the variable that controls batch generation for the
            next epoch to True
        
            Args: None
                
            Returns: None
                
        """
        self.ready_for_next_epoch = True


    def set_stop(self):
        """ Set stop Flag to True
        
            Args: None
                
            Returns: None
                
        """
        self.stop = True


    def set_early_stopping(self):
        """ Set early stopping flag to True
        
            Args: None
                
            Returns: None
                
        """
        self.set_stop()

        
    def generate_batch(self, coords):
        """ Generate one batch of inputs and outputs
            
        """
        
        raise NotImplementedError("Implement this method in a derived class")

    
    def get_negative_batch(self):
        """
            get chrom positions for the negative samples using
            uniform random sampling from across the all chromosomes
            in self.chroms
            
            Returns:
                pandas.DataFrame: dataframe of coordinates 
                    ('chrom' & 'pos')
        """

        # Step 1: select chromosomes, using sampling weights 
        # according to sizes
        chrom_df = self.chrom_sizes_df.sample(
            n=int(self.batch_size*self.negative_sampling_rate),
            weights=self.chrom_sizes_df.weights, replace=True)

        # Step 2: generate 'n' random numbers where 'n' is the length
        # of chrom_df 
        r = [random.random() for _ in range(chrom_df.shape[0])]

        # Step 3. multiply the random numbers with the size column.
        # Additionally, factor in the flank size and jitter while 
        # computing the position
        chrom_df['pos'] = ((chrom_df['size'] - 
                            ((self.input_flank + self.max_jitter)*2)) * r
                           + self.input_flank + self.max_jitter).astype(int)

        return chrom_df[['chrom', 'pos']]

    
    def proc_target(self, coords_df, mpq, proc_idx):
        """
            Function that will be executed in a separate process.
            Takes a dataframe of peak coordinates and parses them in 
            batches, to get one hot encoded sequences and corresponding
            outputs, and adds the batches to the multiprocessing queue.
            Optionally, samples negative locations and adds them to 
            each batch
            
            Args:
                coords_df (pandas.DataFrame): dataframe containing
                    the chrom & peak pos
                mpq (multiprocessing.Queue): The multiprocessing queue
                    to hold the batches
                
            Return
                ---
        """
        
        # divide the coordinates dataframe into batches
        cnt = 0
        for i in range(0, coords_df.shape[0], self.batch_size):   
            # we need to make sure we dont try to fetch 
            # data beyond the length of the dataframe
            if (i + self.batch_size) > coords_df.shape[0]:
                break
                
            batch_df = coords_df.iloc[i:i + self.batch_size]
            
            # add equal number of negative samples
            if self.mode == "train" and \
                self.sampling_mode == 'peaks' and \
                self.negative_sampling_rate > 0.0:
                    
                neg_batch = self.get_negative_batch()
                batch_df = pd.concat([batch_df, neg_batch])
            
            # generate a batch of one hot encoded sequences and 
            # corresponding outputs
            batch = self.generate_batch(batch_df)
            
            # add batch to the multiprocessing queue
            mpq.put(batch)
    
            cnt += 1
        
        logging.debug("{} process {} put {} batches into mpq".format(
            self.mode, proc_idx, cnt))
            
    def stealer(self, mpq, q, num_batches, thread_id):
        """
            Thread target function to "get" (steal) from the
            multiprocessing queue and "put" in the regular queue

            Args:
                mpq (multiprocessing.Queue): The multiprocessing queue
                    to steal from
                q (Queue): The regular queue to put the batch into
                num_batches (int): the number of batches to "steal"
                    from the mp queue
                thread_id (int): thread id for debugging purposes

            Returns:
                ---
        """
        for i in range(num_batches):            
            q.put(mpq.get())

        logging.debug("{} stealer thread {} got {} batches from mpq".format(
            self.mode, thread_id, num_batches))

            
    def epoch_run(self, data):
        """
            Manage batch generation processes & threads
            for one epoch

            Args:
                data (pandas.DataFrame): dataframe with 'chrom' &
                    'pos' columns
                            
            Returns:
                ---
        """
        
        # list of processes that are spawned
        procs = []     
        
        # list of multiprocessing queues corresponding to each 
        # process
        mp_queues = [] 

        # list of stealer threads (that steal the items out of 
        # the mp queues)
        threads = []   
                       
        # the regular queue
        q = Queue()    

        # to make sure we dont flood the user with warning messages
        warning_dispatched = False
        
        # number of data samples to assign to each processor
        samples_per_processor = sequtils.roundToMultiple(
            int(data.shape[0] / self.num_threads), 
            self.batch_size)

        # batches that will be generated by each process thread
        num_batches = []
        
        # spawn processes that will generate batches of data and "put"
        # into the multiprocessing queues
        for i in range(self.num_threads):
            mpq = mp.Queue()

            # give each process a slice of the dataframe of positives
            df = data[i*samples_per_processor : 
                      (i+1)*samples_per_processor][['chrom', 'pos']]

            # the last process gets the leftover data points
            if i == (self.num_threads-1):
                df = pd.concat([df, data[(i+1)*samples_per_processor:]])
                
            num_batches.append(len(df) // self.batch_size)
            
            if df.shape[0] != 0:
                logging.debug("{} spawning process {}, df size {}, "
                              "sum(num_batches) {}".format(
                              self.mode, i, df.shape, sum(num_batches)))

                # spawn and start the batch generation process 
                p = mp.Process(target = self.proc_target, args = [df, mpq, i])
                p.start()
                procs.append(p)
                mp_queues.append(mpq)
                
            else:
                if not warning_dispatched:
                    logging.warn("One or more process threads are not being "
                                 "assigned data for parallel batch "
                                 "generation. You should reduce the number "
                                 "of threads using the --threads option "
                                 "for better performance. Inspect logs for "
                                 "batch assignments.")
                    warning_dispatched = True
                
                logging.debug("{} skipping process {}, df size {}, "
                              "num_batches {}".format(
                              self.mode, i, df.shape, sum(num_batches)))
                
                procs.append(None)
                mp_queues.append(None)

        logging.debug("{} num_batches list {}".format(self.mode, 
                                                      num_batches))
                
        # the threads that will "get" from mp queues 
        # and put into the regular queue
        # this speeds up yielding of batches, because "get"
        # from mp queue is very slow
        for i in range(self.num_threads):
            # start a stealer thread only if data was assigned to
            # the i-th  process
            if num_batches[i] > 0:
                
                logging.debug("{} starting stealer thread {} [{}] ".format(
                    self.mode, i, num_batches[i]))
                
                mp_q = mp_queues[i]
                stealerThread = Thread(target=self.stealer, 
                                       args=[mp_q, q, num_batches[i], i])
                stealerThread.start()
                threads.append(stealerThread)
            else:
                threads.append(None)
                
                logging.debug("{} skipping stealer thread {} ".format(
                    self.mode, i, num_batches))

        return procs, threads, q, sum(num_batches)

    def gen(self):
        """
            generator function to yield batches of data

        """
        
        for i in range(self.epochs):
            # set this flag to False and wait for the
            self.ready_for_next_epoch = False
            
            logging.debug("{} ready set to FALSE".format(self.mode))
            
            if self.shuffle: # shuffle at the beginning of each epoch
                data = self.data.sample(frac = 1.0)
                logging.debug("{} Shuffling complete".format(self.mode))
            else:
                data= self.data

            # spawn multiple processes to generate batches of data in
            # parallel for each epoch
            procs, threads, q, total_batches = self.epoch_run(data)

            logging.debug("{} Batch generation for epoch {} started".format(
                self.mode, i+1))
            
            # yield the correct number of batches for each epoch
            num_skipped = 0
            for j in range(total_batches):      
                batch = q.get()
                if batch is not None:
                    yield batch
                else: 
                    num_skipped += 1

            # wait for batch generation processes to finish once the
            # required number of batches have been yielded
            for j in range(self.num_threads):
                if procs[j] is not None:
                    procs[j].join()
                    
                if threads[j] is not None:
                    threads[j].join()
                
                logging.debug("{} join complete for process {}".format(
                    self.mode, j))
            
            logging.debug("{} Finished join for epoch {}".format(
                self.mode, i+1))
            
            logging.warn("{} batches skipped due to data errors".format(
                num_skipped))
            
            # wait here for the signal 
            while (not self.ready_for_next_epoch) and (not self.stop):
                continue

            logging.debug("{} Ready for next epoch".format(self.mode))
            
            if self.stop:
                logging.debug("{} Terminating batch generation".format(
                    self.mode))
                break


class MBPNetSequenceGenerator(MSequenceGenerator):
    """ Multi task batch data generation for training BPNet
        on high-throughput sequencing data of various
        geonmics assays
    
        Args:
            input_params (dict): python dictionary with information 
                about the input data. Contains the following keys -
                
                "data" (str)
                ------------
                the path to the data directory containing the signal &
                peaks files for each task OR path to json file 
                containing task information. See README for more
                information on how the data directory has to be 
                organized and the format of the json file if you 
                prefer to use a json file
                
                "stranded" (boolean)
                --------------------
                True if data is stranded
                
                "has_control" (boolean)
                -----------------------
                True if control data has been included 
                
            batch_gen_params (dictionary): python dictionaru with batch
                generation parameters. Contains the following keys - 
            
                "input_seq_len" (int)
                ---------------------
                length of input DNA sequence
                
                "output_len" (int)
                ------------------
                length of output profile
                
                "max_jitter" (int)
                ------------------
                maximum value for randomized jitter to offset the peaks
                from the exact center of the input
                
                "rev_comp_aug" (boolean)
                ------------------------
                enable reverse complement augmentation
                
                "negative_sampling_rate" (float)
                --------------------------------
                the fraction of batch_size that determines how many 
                negative samples are added to each batch
            
                "sampling_mode" (str)
                ---------------------
                the mode of sampling chromosome positions - one of
                ['peaks', 'sequential', 'random']
                
                shuffle (boolean)
                -----------------
                specify whether input data is shuffled at the 
                begininning of each epoch
                
                "mode" (str)
                ------------
                "train", "val" or "test"
                 
                "num_positions" (int)
                ---------------------
                specify how many chromosome positions to sample if 
                sampling_mode is "sequential" or "random". Can be 
                omitted if sampling_mode is "peaks", has no effect if
                present.
                 
                "step_size" (int)
                -----------------
                specify step size for sampling chromosome positions if 
                sampling_mode is "sequential". Can be omitted if 
                sampling_mode is "peaks" or "random", has no effect if
                present.
                
            bpnet_params (dictionary): python dictionary containing
                parameters specific to BPNet. Contains the following
                keys - 
                
                "name" (str)
                ------------
                model architecture name
                
                "filters" (int)
                ---------------
                number of filters for BPNet
                
                "control_smoothing" (list)
                --------------------------
                nested list of gaussiam smoothing parameters. Each 
                inner list has two values - [sigma, window_size] for 
                supplemental control tracks
                
            reference_genome (str): the path to the reference genome 
                fasta file
                
            chrom_sizes (str): path to the chromosome sizes file
            
            chroms (str): the list of chromosomes that will be sampled
                for batch generation
                
            num_threads (int): number of parallel threads for batch
                generation
                
            epochs (int): number of iterations for looping over input
                data
                
            batch_size (int): size of each generated batch of data
            
        Attributes:
            name (str): model architecture name
            
            filters (int): number of filters for BPNet
            
            control_smoothing (list): nested list of gaussiam smoothing
                parameters. Each inner list has two values - 
                [sigma, window_size] for supplemental control tracks
        
    """

    def __init__(self, input_params, batch_gen_params, bpnet_params,
                 reference_genome, chrom_sizes, chroms, num_threads=10, 
                 epochs=100, batch_size=64):
        
        # call base class constructor
        super().__init__(input_params, batch_gen_params, reference_genome, 
                         chrom_sizes, chroms, num_threads, epochs, batch_size)
        
        self.stranded = input_params['stranded']
        self.name = bpnet_params['name']
        self.filters = bpnet_params['filters']
        self.control_smoothing = input_params['control_smoothing']

        
    def generate_batch(self, coords):
        """Generate one batch of inputs and outputs for training BPNet
            
            For all coordinates in "coords" fetch sequences &
            one hot encode the sequences. Fetch corresponding
            signal values (for e.g. from a bigwig file). 
            Package the one hot encoded sequences and the output
            values as a tuple.
            
            Args:
                coords (pandas.DataFrame): dataframe with 'chrom' and
                    'pos' columns specifying the chromosome and the 
                    coordinate
                
            Returns:
                tuple: A batch tuple with one hot encoded sequences 
                and corresponding outputs 
        """
        
        # reference file to fetch sequences
        fasta_ref = pyfaidx.Fasta(self.reference)

        # Initialization
        # (batch_size, output_len, 1 + #smoothing_window_sizes)
        control_profile = np.zeros((coords.shape[0], self.output_flank*2, 
                                    1 + len(self.control_smoothing)), 
                                   dtype=np.float32)
        
        # (batch_size)
        control_profile_counts = np.zeros((coords.shape[0]), 
                                          dtype=np.float32)

        # in 'test' mode we only need the sequence & the control
        if self.mode == "train" or self.mode == "val":
            # (batch_size, output_len, #tasks)
            profile = np.zeros((coords.shape[0], self.output_flank*2, 
                                self.num_tasks), dtype=np.float32)
        
            # (batch_size, #tasks)
            profile_counts = np.zeros((coords.shape[0], self.num_tasks), 
                                      dtype=np.float32)
        
        # if reverse complement augmentation is enabled then double the sizes
        if self.mode == "train" and self.rev_comp_aug:
            control_profile = control_profile.repeat(2, axis=0)
            control_profile_counts = control_profile_counts.repeat(2, axis=0)
            profile = profile.repeat(2, axis=0)
            profile_counts = profile_counts.repeat(2, axis=0)
 
        # list of sequences in the batch, these will be one hot
        # encoded together as a single sequence after iterating
        # over the batch
        sequences = []  
        
        # list of chromosome start/end coordinates 
        # useful for tracking test batches
        coordinates = []
        
        # open all the control bigwig files and store the file 
        # objects in a dictionary
        control_files = {}
        for task in self.tasks:
            # the control is not necessary 
            if 'control' in self.tasks[task]:
                control_files[task] = pyBigWig.open(
                    self.tasks[task]['control'])

        # in 'test' mode we only need the sequence & the control
        if self.mode == "train" or self.mode == "val":
            # open all the required bigwig files and store the file 
            # objects in a dictionary
            signal_files = {}
            for task in self.tasks:
                signal_files[task] = pyBigWig.open(self.tasks[task]['signal'])
            
        # iterate over the batch
        rowCnt = 0
        for _, row in coords.iterrows():
            # randomly set a jitter value to move the peak summit 
            # slightly away from the exact center
            jitter = 0
            if self.mode == "train" and self.max_jitter:
                jitter = random.randint(-self.max_jitter, self.max_jitter)
            
            # Step 1 get the sequence 
            chrom = row['chrom']
            # we use self.input_flank here and not self.output_flank because
            # input_seq_len is different from output_len
            start = row['pos'] - self.input_flank + jitter
            end = row['pos'] + self.input_flank + jitter
            seq = fasta_ref[chrom][start:end].seq.upper()
            
            # collect all the sequences into a list
            sequences.append(seq)
            
            start = row['pos'] - self.output_flank  + jitter
            end = row['pos'] + self.output_flank + jitter
            
            # collect all the start/end coordinates into a list
            # we'll send this off along with 'test' batches
            coordinates.append((chrom, start, end))

            # iterate over each task
            for task in self.tasks:
                # identifies the +/- strand pair
                task_id = self.tasks[task]['task_id']
                
                # the strand id: 0-positive, 1-negative
                # easy to index with those values
                strand = self.tasks[task]['strand']
                
                # Step 2. get the control values
                if task in control_files:
                    control_values = control_files[task].values(
                        chrom, start, end)

                    # replace nans with zeros
                    if np.any(np.isnan(control_values)): 
                        control_values = np.nan_to_num(control_values)

                    # update row in batch with the control values
                    # the values are summed across all tasks
                    # the axis = 1 dimension accumulates the sum
                    # there are 'n' copies of the sum along axis = 2, 
                    # n = #smoothing_windows
                    control_profile[rowCnt, :, :] += np.expand_dims(
                        control_values, axis=1)
                
                # in 'test' mode we only need the sequence & the control
                if self.mode == "train" or self.mode == "val":
                    # Step 3. get the signal values
                    # fetch values using the pyBigWig file objects
                    values = signal_files[task].values(chrom, start, end)
                
                    # replace nans with zeros
                    if np.any(np.isnan(values)): 
                        values = np.nan_to_num(values)

                    # update row in batch with the signal values
                    profile[rowCnt, :, task_id*2 + strand] = values

            rowCnt += 1
        
        # Step 4. reverse complement augmentation
        if self.mode == "train" and self.rev_comp_aug:
            # Step 4.1 get list of reverse complement sequences
            rev_comp_sequences = \
                sequtils.reverse_complement_of_sequences(sequences)
            
            # append the rev comp sequences to the original list
            sequences.extend(rev_comp_sequences)
            
            # Step 4.2 reverse complement of the control profile
            control_profile[rowCnt:, :, :] = \
                sequtils.reverse_complement_of_profiles(
                    control_profile[:rowCnt, :, :], self.stranded)
            
            # Step 4.3 reverse complement of the signal profile
            profile[rowCnt:, :, :]  = \
                sequtils.reverse_complement_of_profiles(
                    profile[:rowCnt, :, :], self.stranded)

        # Step 5. one hot encode all the sequences in the batch 
        X = sequtils.one_hot_encode(sequences)

        # if the input sequences are of unequal length then None
        # is returned
        if X is None:
            return None
 
        # we can perform smoothing on the entire batch of control values
        for i in range(len(self.control_smoothing)):
            
                # compute truncate value for scipy gaussian_filter1d
                # "Truncate the filter at this many standard deviations"
                sigma = self.control_smoothing[i][0]
                window_size = self.control_smoothing[i][1]
                truncate = (((window_size - 1)/2)-0.5)/sigma
                
                # its i+1 because at index 0 we have the original 
                # control  
                control_profile[:, :, i+1] = gaussian_filter1d(
                    control_profile[:, :, i+1], sigma=sigma, truncate=truncate)

        # log of sum of control profile without smoothing (idx = 0)
        control_profile_counts = np.log(
            np.sum(control_profile[:, :, 0], axis=-1) + 1)
        
        # in 'test' mode we only need the sequence & the control
        if self.mode == "train" or self.mode == 'val':
            # we can now sum the profiles for the entire batch
            profile_counts = np.log(np.sum(profile, axis=1) + 1)
    
            # return a tuple of input and output dictionaries
            return ({'sequence': X, 
                     'control_profile': control_profile, 
                     'control_logcount': control_profile_counts},
                    {'profile_predictions': profile, 
                     'logcount_predictions': profile_counts})

        # in 'test' mode return a tuple of cordinates & the
        # input dictionary
        return (coordinates, {'sequence': X, 
                            'control_profile': control_profile,
                            'control_logcount': control_profile_counts})
