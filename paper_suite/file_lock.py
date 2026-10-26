'Cross-platform process file lock. Lock file may persist after exit; the OS lock does not.'
from contextlib import contextmanager
import os
import time
from pathlib import Path

@contextmanager
def file_lock(path, *, blocking=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    handle=path.open('a+b')
    if handle.tell()==0:
        handle.write(b'0');handle.flush()
    acquired=False
    try:
        if os.name=='nt':
            import msvcrt
            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                    acquired=True;break
                except OSError:
                    if not blocking:raise RuntimeError('another process holds '+str(path))
                    time.sleep(0.1)
        else:
            import fcntl
            flags=fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB)
            try:fcntl.flock(handle,flags);acquired=True
            except BlockingIOError as exc:raise RuntimeError('another process holds '+str(path)) from exc
        yield handle
    finally:
        if acquired:
            if os.name=='nt':
                import msvcrt
                handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(handle,fcntl.LOCK_UN)
        handle.close()
