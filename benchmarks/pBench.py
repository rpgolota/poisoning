import cProfile, pstats, io
from pstats import SortKey
import timeit
from datetime import timedelta

class pBench_full:
    
    def __init__(self, sort=[SortKey.CUMULATIVE]):
        self.default_sort = sort
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        self.profiler.disable()
    
    def get_stats(self, sort=None):
        if sort is None:
            sort = self.default_sort
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats(*sort)
        pstats.Stats()
        ps.print_stats()
        return s.getvalue()
    
    def show(self, sort=None):
        print(self.get_stats(sort=sort))
        
    def write(self, filename, sort=None, flag='w'):
        with open(filename, flag) as f:
            f.write(self.get_stats(sort=sort))
            
class pBench_fast:
    
    def __init__(self):
        self.current_time = None
        
    def __enter__(self):
        self.current_time = timeit.default_timer()
        return self
    
    def __exit__(self, type, value, traceback):
        self.current_time = timeit.default_timer() - self.current_time
    
    def get_time(self):
        return timedelta(seconds=self.current_time)
    
    def get_seconds(self):
        return self.current_time
    
    def __str__(self):
        return str(self.get_time())
    
    def show(self):
        print(str(self))
        
    def write(self, filename, flag='w'):
        with open(filename, flag) as f:
            f.write(str(self))