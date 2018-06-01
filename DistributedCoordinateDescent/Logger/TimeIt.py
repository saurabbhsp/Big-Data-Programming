import time
import logging

"""
This module provides decorator pattern
helper methods so that they can be used
for logging timing for some functions"""

logger = logging.getLogger('timeLogger')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('timeit.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def timeit(f):
	def timed(*args, **kw):
		ts = time.time()
		result = f(*args, **kw)
		te = time.time()
		logger.info('Function:%r  Execution Time: %2.4f sec' %(f.__name__, te-ts))
		return result
	return timed