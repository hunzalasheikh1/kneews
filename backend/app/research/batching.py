"""Bounded local request concurrency; callers retain single-threaded DB writes."""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED


def concurrent_results(function,items,workers=1):
    if not 1<=workers<=4:raise ValueError('Use one to four local requests')
    source=iter(items);executor=ThreadPoolExecutor(max_workers=workers);pending={}
    def submit():
        try:item=next(source)
        except StopIteration:return
        pending[executor.submit(function,item)]=item
    try:
        for _ in range(workers):submit()
        while pending:
            completed,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in completed:
                item=pending.pop(future)
                try:result=future.result();error=None
                except Exception as exc:result=None;error=exc
                yield item,result,error
                submit()
    finally:
        for future in pending:future.cancel()
        executor.shutdown(wait=True,cancel_futures=True)
