"""
    This module contains classes for all the sequence data generators

    Classes
    
    MSequenceGenerator - The main base class for all generators.
     
    Multi task batch data generation for training deep neural networks
    on high-throughput sequencing data of various geonmics assays
    

    MBPNetSequenceGenerator - Derives from MSequenceGenerator.
    
    Multi task batch data generation for training BPNet on
    high-throughput sequencing data of various geonmics assays
         
    
    IGNORE_FOR_SPHINX_DOCS:
    
    License
    
    MIT License
    
    Copyright (c) 2020 Kundaje Lab
    
    Permission is hereby granted, free of charge, to any person 
    obtaining a copy of this software and associated documentation 
    files (the "Software"), to deal in the Software without 
    restriction, including without limitation the rights to use, 
    copy, modify, merge, publish, distribute, sublicense, and/or 
    sell copiesof the Software, and to permit persons to whom the 
    Software is furnished to do so, subject to the following 
    conditions: 
    
    The above copyright notice and this permission notice shall be 
    included in all copies or substantial portions of the 
    Software. 
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY 
    KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE  
    WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR  
    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS   
    OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR 
    OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR  
    OTHERWISE, ARISING FROM, OUT OF OR IN  CONNECTION WITH THE 
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
    
    
    IGNORE_FOR_SPHINX_DOCS:

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
import re

from mseqgen import sequtils
from mseqgen import quietexception
from mseqgen import utils
from queue import Queue
from threading import Thread


class MSequenceGenerator:
    
    """ Multi task batch data generation for training deep neural
        networks on high-throughput sequencing data of various
        geonmics assays
          
        Args:
            input_config (dict): python dictionary with information 
                about the input data. Contains the following keys -

                *data (str)*
                    path to the json file containing task information. 
                    See README for more information on the format of  
                    the json file
                
                *stranded (boolean)*
                    True if data is stranded
            
                *has_control (boolean)*
                    True if control data has been included 
                
            batch_gen_params (dictionary): python dictionary with batch
                generation parameters. Contains the following keys - 
            
                *input_seq_len (int)*
                    length of input DNA sequence
                
                *output_len (int)*
                    length of output profile
                
                *max_jitter (int)*
                    maximum value for randomized jitter to offset the 
                    peaks from the exact center of the input
                
                *rev_comp_aug (boolean)*
                    enable reverse complement augmentation
                
                *negative_sampling_rate (float)*
                    the fraction of batch_size that determines how many 
                    negative samples are added to each batch
            
                *sampling_mode (str)*
                    the mode of sampling chromosome positions - one of
                    ['peaks', 'sequential', 'random', 'manual']. In 
                    'peaks' mode the data samples are fetched from the
                    peaks bed file specified in the json file 
                    input_config['data']. In 'manual' mode, the two 
                    column pandas dataframe containing the chromosome  
                    position information is passed to the 'samples' 
                    argument of the class
                
                *shuffle (boolean)*
                    specify whether input data is shuffled at the 
                    begininning of each epoch
                
                *mode (str)*
                    'train', 'val' or 'test'
                 
                *num_positions" (int)*
                    specify how many chromosome positions to sample if 
                    sampling_mode is 'sequential' or 'random'. Can be 
                    omitted if sampling_mode is "peaks", has no effect if
                    present.
                 
                *step_size (int)*
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
                generation, default = 10
                
            epochs (int): number of iterations for looping over input
                data, default = 1
                
            batch_size (int): size of each generated batch of data, 
                default = 64
                
            samples (pandas.Dataframe): two column pandas dataframe
                with chromosome position information. Required column
                names are column 1:'chrom', column 2:'pos'. Use this
                parameter if you set batch_gen_params['sampling_mode']
                to 'manual'. default = None

        
        **Members**
        
        IGNORE_FOR_SPHINX_DOCS:
            _stranded (boolean): True if input data is stranded
            
            _has_control (boolean): True if input data includes 
                bias/control track
            
            _sampling_mode (str): the mode of sampling chromosome 
                positions; one of  
                ['peaks', 'sequential', 'random', 'manual'].
            
            _mode (str): 'train', 'val' or 'test'

            _tasks (collections.OrderedDict): dictionary of input tasks
                taken from input_data json
            
            _num_tasks (int): the number of tasks in 'tasks'
            
            _reference (str): the path to the reference genome fasta
                file
            
            _chroms (list): the list of chromosomes that will be sampled
                for batch generation
            
            _chrom_sizes_df (pandas.Dataframe): dataframe of the 
                chromosomes and their corresponding sizes
            
            _num_threads (int): number of parallel threads for batch
                generation
            
            _epochs (int): number of iterations for looping over input
                data
            
            _batch_size (int): size of each generated batch of data
                        
            _input_flank (int): one half of input sequence length
            
            _output_flank (int): one half of output sequence length
            
            _max_jitter (int): the maximum absolute value of jitter to
                vary the position of the peak summit to left or right
                of the exact center of the input sequence. Range is
                -max_jitter to +max_jitter.
            
            _negative_sampling_rate (float): Use a positive value > 0.0
                to specify how many negative samples will be added to
                each batch. num_negative_samples = 
                negative_sampling_rate * batch_size. Ignored if 
                --sampling_mode is not 'peaks', and --mode is not 
                'train'
            
            _rev_comp_aug (boolean): specify whether reverse complement
                augmentation should be applied to each batch of data.
                If True, the size of the generated batch is doubled 
                (i.e batch_size*2 or if negative samples are added then
                (batch_size + num_negative_samples)*2). Ignored if 
                --mode is not 'train'
            
            _shuffle (boolean): if True input data will be shuffled at
                the begininning of each epoch

            _ready_for_next_epoch (boolean): flag to control batch 
                generation for the next epoch. The consumer of the 
                generator is required to send this signal using 
                'set_ready_for_next_epoch'. This protocol is required
                so that excessive and unnecessary batches are not 
                generated if they will not be consumed 
            
            _stop (boolean): flag to indicate that batch generation 
                should be terminated after the current epoch
                
            _samples (pandas.Dataframe): two column pandas dataframe with
                chromosome positions that will be used for generating 
                batches of data
            
        IGNORE_FOR_SPHINX_DOCS
    """

    def __init__(self, input_config, batch_gen_params, reference_genome, 
                 chrom_sizes, chroms, num_threads=10, epochs=1, batch_size=64, 
                 samples=None):
        
        #: True if data is stranded
        self._stranded = input_config['stranded']
        
        #: True if data has controls
        self._has_control = input_config['has_control']
        
        #: ML task mode 'train', 'val' or 'test'
        self._mode = batch_gen_params['mode']

        #: sampling mode to get chromosome positions
        self._sampling_mode = batch_gen_params['sampling_mode']
        
        # make sure the input_data json file exists
        if not os.path.isfile(input_config['data']):
            raise quietexception.QuietException(
                "File not found: {} OR you may have accidentally "
                "specified a directory path.".format(input_config['data']))
        
        # load the json file
        with open(input_config['data'], 'r') as inp_json:
            try:
                #: dictionary of tasks for training
                self._tasks = json.loads(inp_json.read())
            except json.decoder.JSONDecodeError:
                raise quietexception.QuietException(
                    "Unable to load json file {}. Valid json expected. "
                    "Check the file for syntax errors.".format(
                        input_config['data']))

        # check if the reference genome file exists
        if not os.path.isfile(reference_genome):
            raise quietexception.QuietException(
                "File not found: {} OR you may have accidentally "
                "specified a directory path.", reference_genome)
        
        # check if the chrom_sizes file exists
        if not os.path.isfile(chrom_sizes):
            raise quietexception.QuietException(
                "File not found: {} OR you may have accidentally "
                "specified a directory path.".format(chrom_sizes))

        #: the number of tasks in _tasks 
        self._num_tasks = len(list(self._tasks.keys()))
        
        #: path to the reference genome
        self._reference = reference_genome

        #: dataframe of the chromosomes and their corresponding sizes
        self._chrom_sizes_df = pd.read_csv(
            chrom_sizes, sep='\t', header=None, names=['chrom', 'size']) 

        #: list of chromosomes that will be sampled for batch generation
        self._chroms = chroms
        
        # keep only those _chrom_sizes_df rows corresponding to the 
        # required chromosomes in _chroms
        self._chrom_sizes_df = self._chrom_sizes_df[
            self._chrom_sizes_df['chrom'].isin(self._chroms)]

        # generate a new column for sampling weights of the chromosomes
        self._chrom_sizes_df['weights'] = \
            (self._chrom_sizes_df['size'] / self._chrom_sizes_df['size'].sum())

        #: number of parallel threads for batch generation 
        self._num_threads = num_threads
        
        #: number of iterations for looping over input data
        self._epochs = epochs
        
        #: size of each generated batch of data
        self._batch_size = batch_size

        # rest of batch generation parameters
        #: int:one half of input sequence length
        self._input_flank = batch_gen_params['input_seq_len'] // 2
        
        #: one half of input sequence length
        self._output_flank = batch_gen_params['output_len'] // 2        
        
        #: the maximum absolute value of jitter to vary the position
        #: of the peak summit to left or right of the exact center
        #: of the input sequence. Range is -max_jitter to +max_jitter.
        self._max_jitter = batch_gen_params['max_jitter']
        
        #: Use a positive value > 0.0 to specify how many negative
        #: samples will be added to each batch. num_negative_samples = 
        #: negative_sampling_rate * batch_size. Ignored if 
        #: --sampling_mode is not 'peaks', and --mode is not 'train'
        self._negative_sampling_rate = \
            batch_gen_params['negative_sampling_rate']
        
        #: if True, reverse complement augmentation will be applied to
        #: each batch of data. The size of the generated batch is 
        #: doubled (i.e batch_size*2 or if negative samples are added 
        #: then (batch_size + num_negative_samples)*2). Ignored if
        #: --mode is not 'train'
        self._rev_comp_aug = batch_gen_params['rev_comp_aug']
        
        #: if True, shuffle the data before the beginning of the epoch
        self._shuffle = batch_gen_params['shuffle']
        
        if self._sampling_mode == 'peaks':
            # get a pandas dataframe for the peak positions
            # Note - we need the 'tasks' dictionary so we can access
            # the peaks.bed files from the paths available in the 
            # dictionary
            self._samples = sequtils.getPeakPositions(
                self._tasks, self._chroms, 
                self._chrom_sizes_df[['chrom', 'size']], self._input_flank, 
                drop_duplicates=True)
            
        elif self._sampling_mode == 'sequential':
            
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
            self._samples = sequtils.getChromPositions(
                self._chroms, self._chrom_sizes_df[['chrom', 'size']], 
                self._input_flank, mode=self._sampling_mode, 
                num_positions=batch_gen_params['num_positions'], 
                step=batch_gen_params['step_size'])

            # since the positions are fixed and equally spaced we 
            # wont jitter
            self._max_jitter = 0

        elif self._sampling_mode == 'random':
            
            if 'num_positions' not in batch_gen_params:
                raise quietexception.QuietException(
                    "Key not found in batch_gen_params_json: 'num_positions'. "
                    "Required for random sampling mode")
            
            # get a pandas dataframe with random positions
            self._samples = sequtils.getChromPositions(
                self._chroms, self._chrom_sizes_df[['chrom', 'size']], 
                self._input_flank, mode=self._sampling_mode, 
                num_positions=batch_gen_params['num_positions'])

            # its already random, why jitter?!
            self._max_jitter = 0
        
        elif self._sampling_mode == 'manual':
            
            # check if the samples parameter has been provided
            if samples is None:
                raise quietexception.QuietException(
                    "If sampling_mode is 'manual', 'samples' parameter"
                    "has to be set. Found None.")
            
            if not isinstance(samples, pandas.Dataframe) or \
                    set(samples.columns.tolist()) != set(['chrom', 'pos']):
                raise quietexception.QuietException(
                    "samples' parameter should be a valid pandas.Dataframe"
                    "with two columns 'chrom' and 'pos'")
                
            #: two column pandas dataframe with chromosome positions,
            #: columns = ['chrom', 'pos']
            self._samples = samples
            
        #: size of the input samples before padding
        self._unpadded_samples_size = len(self._samples)
        
        # pad self._samples dataframe with randomly selected rows 
        # so that the length of the dataframe is an exact multiple of
        # num_threads * batch_size. We do this so we can equally divide
        # the batches across several batch generation threads 
        exact_multiple = sequtils.round_to_multiple(
            len(self._samples), num_threads * batch_size, smallest=True)
        pad_size = exact_multiple - len(self._samples)
        
        if pad_size > 0:
            # If the pad_size > #self._samples, then number of data
            # samples for the set (train or val) is significantly less
            # than num_threads * batch_size, so we'll have to sample 
            # the padded rows with replacement
            replace = False
            if pad_size > len(self._samples):
                replace = True
                logging.info("mode '{}': Sampling with replacement for "
                             "data padding")
            
            self._samples = self._samples.append(
                self._samples.sample(pad_size, replace=replace),
                ignore_index=True)

        #: size of the input samples after padding
        self._samples_size = len(self._samples)
        
        logging.info("mode '{}': Data size (with {} padded rows) - {}".format(
            self._mode, pad_size, len(self._samples)))
        
    def get_input_tasks(self):
        """
            The dictionary of tasks loaded from the json file
            input_config['data']
            
            Returns:
                
                dict: dictionary of input tasks
        """
        
        return self._tasks
    
    def get_unpadded_samples_len(self):
        """
            The number of data samples before padding
            
            Returns:
                
                int: number of data samples before padding
        """
        
        return self._unpadded_samples_size
    
    def get_samples_len(self):
        """
            The number of data samples used in batch generation
            (after padding)
            
            Returns:
                
                int: number of data samples used in batch generation
        """
        
        return self._samples_size
    
    def len(self):
        """
            The number of batches per epoch
            
            Returns:
                int: number of batches of data generated in each epoch
        """
        
        return self._samples.shape[0] // self._batch_size
   
    def _generate_batch(self, coords):
        """ 
            Generate one batch of inputs and outputs
            
        """
        
        raise NotImplementedError("Method not implemented. Used a "
                                  "derived class.")

    def get_name(self):
        """ 
            Name of the sequence generator
            
        """
        raise NotImplementedError("Method not implemented. Used a "
                                  "derived class.")

    def _get_negative_batch(self):
        """
            Get chrom positions for the negative samples using
            uniform random sampling from across the all chromosomes
            in self._chroms
            
            Returns:
                pandas.DataFrame: 
                    two column dataframe of chromosome positions with
                    'chrom' & 'pos' columns

        """

        # Step 1: select chromosomes, using sampling weights 
        # according to sizes
        chrom_df = self._chrom_sizes_df.sample(
            n=int(self._batch_size * self._negative_sampling_rate),
            weights=self._chrom_sizes_df.weights, replace=True)

        # Step 2: generate 'n' random numbers where 'n' is the length
        # of chrom_df 
        r = [random.random() for _ in range(chrom_df.shape[0])]

        # Step 3. multiply the random numbers with the size column.
        # Additionally, factor in the flank size and jitter while 
        # computing the position
        chrom_df['pos'] = ((chrom_df['size'] - ((self._input_flank
                                                 + self._max_jitter) * 2))
                           * r + self._input_flank
                           + self._max_jitter).astype(int)

        return chrom_df[['chrom', 'pos']]

    def _proc_target(self, coords_df, mpq, proc_idx):
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
        """
        
        # divide the coordinates dataframe into batches
        cnt = 0
        for i in range(0, coords_df.shape[0], self._batch_size):   
            # we need to make sure we dont try to fetch 
            # data beyond the length of the dataframe
            if (i + self._batch_size) > coords_df.shape[0]:
                break
                
            batch_df = coords_df.iloc[i:i + self._batch_size]
            batch_df = batch_df.copy()
            batch_df['status'] = 1
            
            # add equal number of negative samples
            if self._mode == "train" and \
                self._sampling_mode == 'peaks' and \
                    self._negative_sampling_rate > 0.0:
                    
                neg_batch = self._get_negative_batch()
                neg_batch['status'] = -1
                batch_df = pd.concat([batch_df, neg_batch])
            
            # generate a batch of one hot encoded sequences and 
            # corresponding outputs
            batch = self._generate_batch(batch_df)
            
            # add batch to the multiprocessing queue
            mpq.put(batch)
    
            cnt += 1
        
        logging.debug("{} process {} put {} batches into mpq".format(
            self._mode, proc_idx, cnt))
            
    def _stealer(self, mpq, q, num_batches, thread_id):
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

        """
        for i in range(num_batches):            
            q.put(mpq.get())

        logging.debug("{} stealer thread {} got {} batches from mpq".format(
            self._mode, thread_id, num_batches))

    def _epoch_run(self, data):
        """
            Manage batch generation processes & threads
            for one epoch

            Args:
                data (pandas.DataFrame): dataframe with 'chrom' &
                    'pos' columns
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
        # (since we have already padded data len(data) is directly
        # divisible by num_threads)
        samples_per_processor = int(len(data) / self._num_threads)

        # batches that will be generated by each process thread
        num_batches = []
        
        # spawn processes that will generate batches of data and "put"
        # into the multiprocessing queues
        for i in range(self._num_threads):
            mpq = mp.Queue()

            # give each process a slice of the dataframe of positives
            df = data[i * samples_per_processor: 
                      (i + 1) * samples_per_processor][['chrom', 'pos']]

            # the last process gets the leftover data points
            if i == (self._num_threads - 1):
                df = pd.concat([df, data[(i + 1) * samples_per_processor:]])
                
            num_batches.append(len(df) // self._batch_size)
            
            if df.shape[0] != 0:
                logging.debug("{} spawning process {}, df size {}, "
                              "sum(num_batches) {}".format(
                                  self._mode, i, df.shape, sum(num_batches)))

                # spawn and start the batch generation process 
                p = mp.Process(target=self._proc_target, args=[df, mpq, i])
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
                                  self._mode, i, df.shape, sum(num_batches)))
                
                procs.append(None)
                mp_queues.append(None)

        logging.debug("{} num_batches list {}".format(self._mode, 
                                                      num_batches))
                
        # the threads that will "get" from mp queues 
        # and put into the regular queue
        # this speeds up yielding of batches, because "get"
        # from mp queue is very slow
        for i in range(self._num_threads):
            # start a stealer thread only if data was assigned to
            # the i-th  process
            if num_batches[i] > 0:
                
                logging.debug("{} starting stealer thread {} [{}] ".format(
                    self._mode, i, num_batches[i]))
                
                mp_q = mp_queues[i]
                stealerThread = Thread(target=self._stealer, 
                                       args=[mp_q, q, num_batches[i], i])
                stealerThread.start()
                threads.append(stealerThread)
            else:
                threads.append(None)
                
                logging.debug("{} skipping stealer thread {} ".format(
                    self._mode, i, num_batches))

        return procs, threads, q, sum(num_batches)

    def gen(self, epoch):
        """
            Generator function to yield one batch of data
            
            Args:
                epoch (int): the epoch number

        """
        
        if self._shuffle:
            # shuffle at the beginning of each epoch
            data = self._samples.sample(frac=1.0)
            logging.debug("{} Shuffling complete".format(self._mode))
        else:
            data = self._samples

        # spawn multiple processes to generate batches of data in
        # parallel for each epoch
        procs, threads, q, total_batches = self._epoch_run(data)

        logging.debug("{} Batch generation for epoch {} started".format(
            self._mode, epoch))

        # yield the correct number of batches for each epoch
        for j in range(total_batches):      
            batch = q.get()
            yield batch

        # wait for batch generation processes to finish once the
        # required number of batches have been yielded
        for j in range(self._num_threads):
            if procs[j] is not None:
                logging.debug("{} waiting to join process {}".format(
                    self._mode, j))
                procs[j].join()

            if threads[j] is not None:
                logging.debug("{} waiting to join thread {}".format(
                    self._mode, j))
                threads[j].join()

            logging.debug("{} join complete for process {}".format(
                self._mode, j))

        logging.debug("{} Finished join for epoch {}".format(
            self._mode, epoch))

        logging.debug("{} Ready for next epoch".format(self._mode))


class MBPNetSequenceGenerator(MSequenceGenerator):
    """ 
        Multi task batch data generation for training BPNet
        on high-throughput sequencing data of various
        geonmics assays
    
        Args:
            input_config (dict): python dictionary with information 
                about the input data. Contains the following keys -

                *data (str)*
                    path to the json file containing task information. 
                    See README for more information on the format of  
                    the json file
                
                *stranded (boolean)*
                    True if data is stranded
            
                *has_control (boolean)*
                    True if control data has been included 
                
            batch_gen_params (dictionary): python dictionary with batch
                generation parameters. Contains the following keys - 
            
                *input_seq_len (int)*
                    length of input DNA sequence
                
                *output_len (int)*
                    length of output profile
                
                *max_jitter (int)*
                    maximum value for randomized jitter to offset the 
                    peaks from the exact center of the input
                
                *rev_comp_aug (boolean)*
                    enable reverse complement augmentation
                
                *negative_sampling_rate (float)*
                    the fraction of batch_size that determines how many 
                    negative samples are added to each batch
            
                *sampling_mode (str)*
                    the mode of sampling chromosome positions - one of
                    ['peaks', 'sequential', 'random', 'manual']. In 
                    'peaks' mode the data samples are fetched from the
                    peaks bed file specified in the json file 
                    input_config['data']. In 'manual' mode, the bed
                    file containing the chromosome position information
                    is passed to the 'samples' argument of the class
                
                *shuffle (boolean)*
                    specify whether input data is shuffled at the 
                    begininning of each epoch
                
                *mode (str)*
                    'train', 'val' or 'test'
                 
                *num_positions" (int)*
                    specify how many chromosome positions to sample if 
                    sampling_mode is 'sequential' or 'random'. Can be 
                    omitted if sampling_mode is "peaks", has no effect if
                    present.
                 
                *step_size (int)*
                    specify step size for sampling chromosome positions if 
                    sampling_mode is "sequential". Can be omitted if 
                    sampling_mode is "peaks" or "random", has no effect if
                    present.

            bpnet_params (dictionary): python dictionary containing
                parameters specific to BPNet. Contains the following
                keys - 
                
                *name (str)*
                    model architecture name
                
                *filters (int)*
                    number of filters for BPNet
                
                *control_smoothing (list)*
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

            samples (pandas.Dataframe): two column pandas dataframe
                with chromosome position information. Required column
                names are column 1:'chrom', column 2:'pos'. Use this
                parameter if you set batch_gen_params['sampling_mode']
                to 'manual'. default = None

        **Members**
        
        IGNORE_FOR_SPHINX_DOCS:
        Attributes:
            _control_smoothing (list): nested list of gaussiam smoothing
                parameters. Each inner list has two values - 
                [sigma, window_size] for supplemental control tracks
        IGNORE_FOR_SPHINX_DOCS
        
    """

    def __init__(self, input_config, batch_gen_params, bpnet_params,
                 reference_genome, chrom_sizes, chroms, num_threads=10, 
                 epochs=100, batch_size=64, samples=None):
        
        # name of the generator class
        self.name = "BPNet"
        
        # call base class constructor
        super().__init__(input_config, batch_gen_params, reference_genome, 
                         chrom_sizes, chroms, num_threads, epochs, batch_size, 
                         samples)
        
        #: nested list of gaussiam smoothing parameters. Each inner list
        #: has two values - [sigma, window_size] for supplemental
        #: control control tracks
        self._control_smoothing = bpnet_params['control_smoothing']

    def _generate_batch(self, coords):
        """Generate one batch of inputs and outputs for training BPNet
            
            For all coordinates in "coords" fetch sequences &
            one hot encode the sequences. Fetch corresponding
            signal values (for e.g. from a bigwig file). 
            Package the one hot encoded sequences and the output
            values as a tuple.
            
            Args:
                coords (pandas.DataFrame): dataframe with 'chrom', 
                    'pos' & 'status' columns specifying the chromosome,
                    thecoordinate and whether the loci is a positive(1)
                    or negative sample(-1)
                
            Returns:
                tuple: 
                    When 'mode' is 'train' or 'val' a batch tuple 
                    with one hot encoded sequences and corresponding 
                    outputs and when 'mode' is 'test' tuple of 
                    cordinates & the inputs
        """
        
        # reference file to fetch sequences
        fasta_ref = pyfaidx.Fasta(self._reference)

        # Initialization
        # (batch_size, output_len, 1 + #smoothing_window_sizes)
        control_profile = np.zeros((coords.shape[0], self._output_flank * 2, 
                                    1 + len(self._control_smoothing)), 
                                   dtype=np.float32)
        
        # (batch_size)
        control_profile_counts = np.zeros((coords.shape[0]), 
                                          dtype=np.float32)

        # in 'test' mode we pass the true profile as part of the 
        # returned tuple from the batch generator
        if self._mode == "train" or self._mode == "val" or \
                self._mode == "test":
            # (batch_size, output_len, #tasks)
            profile = np.zeros((coords.shape[0], self._output_flank * 2, 
                                self._num_tasks), dtype=np.float32)

            # (batch_size, #tasks)
            profile_counts = np.zeros((coords.shape[0], self._num_tasks),
                                      dtype=np.float32)
        
        # if reverse complement augmentation is enabled then double the sizes
        if self._mode == "train" and self._rev_comp_aug:
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
        for task in self._tasks:
            # the control is not necessary 
            if 'control' in self._tasks[task]:
                control_files[task] = pyBigWig.open(
                    self._tasks[task]['control'])

        # in 'test' mode we pass the true profile as part of the 
        # returned tuple from the batch generator
        if self._mode == "train" or self._mode == "val" or \
                self._mode == "test":
            # open all the required bigwig files and store the file 
            # objects in a dictionary
            signal_files = {}
            for task in self._tasks:
                signal_files[task] = pyBigWig.open(self._tasks[task]['signal'])
            
        # iterate over the batch
        rowCnt = 0
        for _, row in coords.iterrows():
            # randomly set a jitter value to move the peak summit 
            # slightly away from the exact center
            jitter = 0
            if self._mode == "train" and self._max_jitter:
                jitter = random.randint(-self._max_jitter, self._max_jitter)
            
            # Step 1 get the sequence 
            chrom = row['chrom']
            # we use self._input_flank here and not self._output_flank because
            # input_seq_len is different from output_len
            start = row['pos'] - self._input_flank + jitter
            end = row['pos'] + self._input_flank + jitter
            seq = fasta_ref[chrom][start:end].seq.upper()
            
            # collect all the sequences into a list
            sequences.append(seq)
            
            start = row['pos'] - self._output_flank + jitter
            end = row['pos'] + self._output_flank + jitter
            
            # collect all the start/end coordinates into a list
            # we'll send this off along with 'test' batches
            coordinates.append((chrom, start, end))

            # iterate over each task
            for task in self._tasks:
                # identifies the +/- strand pair
                task_id = self._tasks[task]['task_id']
                
                # the strand id: 0-positive, 1-negative
                # easy to index with those values
                strand = self._tasks[task]['strand']
                
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
                
                # in 'test' mode we pass the true profile as part of the 
                # returned tuple from the batch generator
                if self._mode == "train" or self._mode == "val" or \
                        self._mode == "test":
                    # Step 3. get the signal values
                    # fetch values using the pyBigWig file objects
                    values = signal_files[task].values(chrom, start, end)
                
                    # replace nans with zeros
                    if np.any(np.isnan(values)): 
                        values = np.nan_to_num(values)

                    # update row in batch with the signal values
                    if self._stranded:
                        profile[rowCnt, :, task_id * 2 + strand] = values
                    else:
                        profile[rowCnt, :, task_id] = values

            rowCnt += 1
        
        # Step 4. reverse complement augmentation
        if self._mode == "train" and self._rev_comp_aug:
            # Step 4.1 get list of reverse complement sequences
            rev_comp_sequences = \
                sequtils.reverse_complement_of_sequences(sequences)
            
            # append the rev comp sequences to the original list
            sequences.extend(rev_comp_sequences)
            
            # Step 4.2 reverse complement of the control profile
            control_profile[rowCnt:, :, :] = \
                sequtils.reverse_complement_of_profiles(
                    control_profile[:rowCnt, :, :], self._stranded)
            
            # Step 4.3 reverse complement of the signal profile
            profile[rowCnt:, :, :] = \
                sequtils.reverse_complement_of_profiles(
                    profile[:rowCnt, :, :], self._stranded)

        # Step 5. one hot encode all the sequences in the batch 
        if len(sequences) == profile.shape[0]:
            X = sequtils.one_hot_encode(sequences, self._input_flank * 2)
        else:
            raise quietexception.QuietException(
                "Unable to generate enough sequences for the batch")
                
        # we can perform smoothing on the entire batch of control values
        for i in range(len(self._control_smoothing)):

            sigma = self._control_smoothing[i][0]
            window_size = self._control_smoothing[i][1]

            # its i+1 because at index 0 we have the original 
            # control  
            control_profile[:, :, i + 1] = utils.gaussian1D_smoothing(
                control_profile[:, :, i + 1], sigma, window_size)

        # log of sum of control profile without smoothing (idx = 0)
        control_profile_counts = np.log(
            np.sum(control_profile[:, :, 0], axis=-1) + 1)
        
        # in 'train' and 'val' mode we need input and output 
        # dictionaries
        if self._mode == "train" or self._mode == 'val':
            # we can now sum the profiles for the entire batch
            profile_counts = np.log(np.sum(profile, axis=1) + 1)
    
            # return a tuple of input and output dictionaries
            # 'coordinates' and 'status are not inputs to the model,
            # so you will see a warning about unused inputs while
            # training. It's safe to ignore the warning
            # We pass 'coordinates' so we can track the exact
            # coordinates of the inputs (because jitter is random)
            # 'status' refers to whether the data sample is a +ve (1)
            # or -ve (-1) example and is used by the attribution
            # prior loss function
            return ({'coordinates': coordinates,
                     'status': coords['status'].values,
                     'sequence': X, 
                     'control_profile': control_profile, 
                     'control_logcount': control_profile_counts},
                    {'profile_predictions': profile, 
                     'logcount_predictions': profile_counts})

        # in 'test' mode return a tuple of cordinates, true profiles
        # & the input dictionary
        return (coordinates, profile,
                {'sequence': X, 
                 'control_profile': control_profile,
                 'control_logcount': control_profile_counts})

       
def list_generator_names():
    """
       List all available sequence generators that are derived
       classes of the base class MSequenceGenerator
       
       Returns:
           list: list of sequence generator names
    """
    
    generator_names = []
    for c in MSequenceGenerator.__subclasses__():        
        result = re.search('M(.*)SequenceGenerator', c.__name__)
        generator_names.append(result.group(1))

    return generator_names


def find_generator_by_name(generator_name):
    """
        Get the sequence generator class name given its name
        
        Returns:
            str: sequence generator class name
    """
    
    for c in MSequenceGenerator.__subclasses__():
        result = re.search('M(.*)SequenceGenerator', c.__name__)
        if generator_name == result.group(1):
            return c.__name__
