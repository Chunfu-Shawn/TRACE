import time
import multiprocessing
import numpy as np
import pickle
from numba import njit
mp_method = 'fork' if 'fork' in multiprocessing.get_all_start_methods() else 'spawn'
mp_ctx = multiprocessing.get_context(mp_method)
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from intervaltree import IntervalTree
import pysam
from data.transcript_exon_index import convert_position

__author__ = "Chunfu Xiao"
__contributor__="..."
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__="1.3.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@126.com"

def zero():
    return 0

def nested_zero_defaultdict():
    return defaultdict(zero)

def double_nested_zero_defaultdict():
    return defaultdict(nested_zero_defaultdict)


@njit(cache=True)
def is_compatible(starts, ends, exon_starts0, exon_ends0, tol=2):
    """
    Junction-reads boundary tolerance check:
    - blocks[0].start must be inside the exon (>= exon_start - tol)
    - blocks[-1].end must be inside the exon (<= exon_end + tol)
    - middle blocks must strictly align to boundaries (±tol)
    - exon order must be contiguous (no gaps)
    """

    # 1) calculate gaps to find real breaks of alignment, excluding the case of CIGAR: M1DM
    if len(starts) > 1:
        gaps = starts[1:] - ends[:-1]
        breaks_idx = np.nonzero(gaps > tol)[0]
        # first and last block indices of each segment
        seg_starts = np.concatenate((
            np.array([0], dtype=breaks_idx.dtype),
            breaks_idx + 1
        ))
        seg_ends   = np.concatenate((
            breaks_idx,
            np.array([len(starts) - 1], dtype=breaks_idx.dtype)
        ))
    else:
        seg_starts = np.array([0])
        seg_ends   = np.array([0])

    # merged blocks ignoring 2 nt gap
    m_starts = starts[seg_starts]
    m_ends = ends[seg_ends]
    B = len(m_starts)

    # 2) binary search: find which exon each block belongs to (with tol)
    idxs = np.searchsorted(exon_starts0 - tol, m_starts , side='right') - 1
    if np.any(idxs < 0):
        return False

    # 3) middle block starts/ends (if B>2) must satisfy boundary alignment
    if B > 2:
        starts_without_first = m_starts[1:]
        ends_without_last = m_ends[:-1]
        
        # start or end alignment (±tol)
        start_ok = np.abs(starts_without_first - exon_starts0[idxs[1:]]) <= tol
        end_ok = np.abs(ends_without_last - exon_ends0[idxs[:-1]]) <= tol
        if not np.all(start_ok & end_ok):
            # print("Junction Error", idxs, m_starts, exon_starts0, m_ends, exon_ends0)
            return False

    # 4) exon-order contiguity check
    diffs = np.diff(idxs)
    if np.any(diffs < 0) or np.any(diffs > 1):
        # print("Exon-order Error", idxs, diffs, m_starts, exon_starts0, m_ends, exon_ends0)
        return False

    return True


class RPF_Counter:
    def __init__(self, 
                 chroms: list, 
                 tree_index_file: str, 
                 tx_meta_file: str, 
                 min_readlen: int = 25, 
                 max_readlen: int = 34, 
                 tol: int = 2):
        # load optimized index
        self.chroms = chroms
        self.tree_index_file = tree_index_file
        self.tx_meta_file = tx_meta_file
        with open(self.tx_meta_file, 'rb') as f:
            tx_meta = pickle.load(f)

        # length of chrom
        self.chrom_lengths = {
            chrom: max(meta['exon_ends0_sorted'][-1] for meta in tx_meta.values() if meta['chrom']==chrom)
            for chrom in self.chroms
        }
        self.min_readlen = int(min_readlen)
        self.max_readlen = int(max_readlen)
        self.tol = int(tol)

    def save_count(self, final_counts, count_file):
        ''' save count data as .pkl file'''
        # save data by pickle
        with open(count_file, 'wb') as f_RPF:
            pickle.dump(final_counts, f_RPF, protocol=pickle.HIGHEST_PROTOCOL)
    
    def process_window(self, args):
        chrom, start, end, tid_list, bam_path = args # , tree_index, tx_meta
        worker_start_time = time.time()
        print("### " + chrom + " ###")

        counts = defaultdict(double_nested_zero_defaultdict)

        # load data
        with open(self.tree_index_file, 'rb') as f:
            tree_index = pickle.load(f)
            
        # strand-specific
        if not tid_list:
            filterd_tree_index = tree_index[chrom]
        else:
            filterd_tree_index = {
                "+": IntervalTree([iv for iv in tree_index[chrom]["+"] if iv.data in tid_list]),
                "-": IntervalTree([iv for iv in tree_index[chrom]["-"] if iv.data in tid_list])
            }

        with open(self.tx_meta_file, 'rb') as f:
            tx_meta = pickle.load(f)

        print(f'Executor {chrom}:{start}-{end} started')

        # window-specific reads
        with pysam.AlignmentFile(bam_path, 'rb', threads=5) as bam_cache:
            for read in bam_cache.fetch(chrom, start-1, end):
                blk = np.array(read.get_blocks(), dtype=int)
                if blk.size == 0:
                    continue
            
                # reasonable RPF length
                read_len = read.query_length
                read_strand = "-" if read.is_reverse else "+"
                if read_len is None or read_len < self.min_readlen or read_len > self.max_readlen:
                    continue
            
                # 5'end and 3'end genomic position, 0-base to 1-base
                starts = blk[:,0]
                ends   = blk[:,1]
                left_prime = starts[0] + 1
                right_prime = ends[-1]
                five_prime = right_prime if read.is_reverse else left_prime

                # find all transcripts overlapping reads
                cand = set(iv.data for iv in filterd_tree_index[read_strand][five_prime])
                # & set(iv.data for iv in filterd_tree_index[right_prime])
    
                # transfer genomic position to tx position
                for tid in cand:
                    meta = tx_meta[tid]
                    # compatible with transcript exon structure ?
                    if not is_compatible(starts, ends,
                                         meta['exon_starts0_sorted'],
                                         meta['exon_ends0_sorted'],
                                         self.tol):
                        continue
                    # transfer to transcript position (input 1-based)
                    pos = convert_position(
                        five_prime,
                        meta['exon_starts'], meta['exon_ends'],
                        meta['tx_starts'], meta['tx_ends'],
                        meta['strand']
                    )
                    # count the read
                    if pos >= 1:
                        counts[tid][pos][read_len] += 1
                        
        print(f'Executor {chrom}:{start}-{end} elapsed time: {time.time() - worker_start_time} seconds')
        return counts

    def parallel_count_by_windows(self, 
                                  bam_path, 
                                  tid_list: list = None,
                                  window_size: int = 20000000, 
                                  max_workers: int = 20):
        start_time = time.time()
        tid_list = [] if tid_list is None else tid_list
        
        """ process all windows parallelly """
        # create tasks
        print("--- Create parallel tasks ---")
        tasks = []
        for chrom, length in self.chrom_lengths.items():
            for start in range(1, length+1, window_size):
                end = min(start + window_size - 1, length)
                tasks.append((chrom, start, end, tid_list, bam_path))
        end_time = time.time()
        print(f'exec time: {end_time - start_time} seconds')

        # 2) parallel processing
        final_counts = defaultdict(double_nested_zero_defaultdict)
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as exe:
            futures = {exe.submit(self.process_window, task): task for task in tasks} # tasks[::-1] reverse chromosomes for saving time
            
            for future in as_completed(futures):
                result = future.result()
                # combine results
                for tid, position_data in result.items():
                    for position, length_data in position_data.items():
                        for read_length, count in length_data.items():
                            final_counts[tid][position][read_length] += count
        return dict(final_counts)
