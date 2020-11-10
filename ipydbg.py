import sys, inspect, traceback

selectable = False
_original_excepthook = None

def hook(exc_type, exc, tb):
    global selectable
    
    from IPython import embed
    from IPython.core.ultratb import VerboseTB

    tb_handler = VerboseTB(color_scheme='Neutral')
    tb_handler(exc_type, exc, tb)
    
    frames = [t[0] for t in traceback.walk_tb(tb)][::-1]
    
    while True:
        print()
        
        if selectable:
            print('Select a stack frame to embed IPython shell:')
            for i, frame in enumerate(frames):
                print('{}. {}'.format(i, frame))
            try:
                s = input('> ').strip()
                n = int(s) if s else 0
                frame = frames[n]
            except (KeyboardInterrupt, EOFError):
                break
            except:
                continue
        else:
            frame = frames[0]

        print('Embedded into', frame)
        
        user_module = inspect.getmodule(frame)
        user_ns = frame.f_locals
        
        user_ns.setdefault('etype', exc_type)
        user_ns.setdefault('evalue', exc)
        user_ns.setdefault('etb', tb)
    
        embed(banner1='', user_module=user_module, user_ns=user_ns, colors='Neutral')
        
        if not selectable:
            break

def patch():
    global _original_excepthook
    assert sys.excepthook is not hook
    _original_excepthook = sys.excepthook
    sys.excepthook = hook

def unpatch():
    global _original_excepthook
    assert _original_excepthook is hook
    sys.excepthook = _original_excepthook


if __name__ == '__main__':
    def test():
        n = 0
        print('hello:', 1 / n)

    patch()
    test()
    
