import threading
from app.research.batching import concurrent_results


def test_concurrency_is_bounded_and_failures_preserve_input_identity():
    barrier=threading.Barrier(2)
    active=0;peak=0;lock=threading.Lock()
    def task(value):
        nonlocal active,peak
        with lock:active+=1;peak=max(peak,active)
        barrier.wait(timeout=5)
        with lock:active-=1
        if value==2:raise ValueError('fixture failure')
        return value*10
    results=list(concurrent_results(task,range(4),workers=2))
    assert peak==2 and {item for item,_,_ in results}=={0,1,2,3}
    assert all(result==item*10 for item,result,error in results if error is None)
    failed=[item for item,_,error in results if isinstance(error,ValueError)]
    assert failed==[2]
