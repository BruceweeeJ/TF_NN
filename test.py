def isPal(s):
    if len(s) <= 1:
        return True
    else:
        return s[0] == s[-1] and isPal(s[1:-1])

def prin(s):
    if len(s) <= 1:
        return True
    print(s[1:-1])
    return prin(s[1:-1])
s = 'doggod'
prin(s)
