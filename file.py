import sys

prime = 16777259
shares = [
    "018c5b2a",
    "0261863a",
    "03e27621",
    "05ccf1ab",
    "060ada60"
]

points = []
for s in shares:
    s = s.strip()
    if len(s) == 7:
        s = '0' + s
    b = bytes.fromhex(s)
    x = b[0]
    y = int.from_bytes(b[1:], 'big')
    points.append((x, y))

def _modinv(a, p):
    a %= p
    if a == 0: raise ZeroDivisionError()
    t, new_t = 0, 1
    r, new_r = p, a
    while new_r != 0:
        q = r // new_r
        t, new_t = new_t, t - q * new_t
        r, new_r = new_r, r - q * new_r
    if t < 0: t += p
    return t

secret = 0
for i in range(len(points)):
    x_i, y_i = points[i]
    numer = 1
    denom = 1
    for j in range(len(points)):
        if i == j: continue
        x_j, _ = points[j]
        numer = (numer * (-x_j % prime)) % prime
        denom = (denom * ((x_i - x_j) % prime)) % prime
    li0 = (numer * _modinv(denom, prime)) % prime
    secret = (secret + (y_i % prime) * li0) % prime

secret &= 0xFFFFFF
print(f"Reconstructed color: #{secret:06X}")