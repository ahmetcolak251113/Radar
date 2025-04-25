import numpy as np

def ucgen(x, abc):
    # a <= b <= c olmalıdır.
    assert len(abc) == 3, "Başlangıç ve bitiş değerleri 3 tane olmalıdır."
    a, b, c = np.r_[abc]
    assert a <= b <= c, "Üyelik fonksiyon değerleri başlangıç <= Tepe <= bitiş"
    y = np.zeros(len(x))
    # sol kenar
    if a != b:
        idx = np.nonzero(np.logical_and(a < x, x < b))[0]
        y[idx] = (x[idx] - a) / float(b - a)
    # sağ kenar
    if b != c:
        idx = np.nonzero(np.logical_and(b < x, x < c))[0]
        y[idx] = (c - x[idx]) / float(c - b)
    # tepe noktası
    idx = np.nonzero(x == b)[0]
    y[idx] = 1.0
    return y

def trapez(x, rot, abc):
    y = np.zeros(len(x))
    # ORTA trapezoid: yükselen kenar + düz plato
    if rot == "ORTA":
        assert len(abc) == 3, "Başlangıç ve bitiş değerleri 3 tane olmalıdır."
        a, b, c = np.r_[abc]
        assert a <= b <= c, "a <= b <= c olmalı"
        # yükselen kenar (a ≤ x < b)
        idx = np.nonzero(np.logical_and(x >= a, x < b))[0]
        y[idx] = (x[idx] - a) / float(b - a)
        # düz plato (b ≤ x ≤ c)
        idx = np.nonzero(np.logical_and(x >= b, x <= c))[0]
        y[idx] = 1.0
        return y

    # SOL trapezoid: plato solda + inen kenar
    assert len(abc) == 2, "Başlangıç ve bitiş değerleri 2 tane olmalıdır."
    a, b = np.r_[abc]
    assert a <= b, "a <= b olmalı"
    if rot == "SOL":
        # plato bölgesi (x ≤ a)
        idx = np.nonzero(x <= a)[0]
        y[idx] = 1.0
        # inen kenar (a < x < b)
        idx = np.nonzero(np.logical_and(x > a, x < b))[0]
        y[idx] = (b - x[idx]) / float(b - a)
        return y

    # SAĞ trapezoid: artan kenar + plato sağda
    if rot == "SAĞ":
        # artan kenar (a < x < b)
        idx = np.nonzero(np.logical_and(x > a, x < b))[0]
        y[idx] = (x[idx] - a) / float(b - a)
        # plato bölgesi (x ≥ b)
        idx = np.nonzero(x >= b)[0]
        y[idx] = 1.0
        return y

    # eğer rot parametresi beklenmeyen bir şeyse
    raise ValueError("rot parametresi 'ORTA', 'SOL' veya 'SAĞ' olmalı")

def uyelik(x, xmf, xx, zero_outside_x = True):
    if not zero_outside_x:
        kwargs = (None, None)
    else:
        kwargs = (0.0, 0.0)
    # Numpy interpolasyon fonksiyonu
    return np.interp(xx, x, xmf, left=kwargs[0], right=kwargs[1])

"""
Üyelik değerlerini hesaplayan fonksiyonumuz basitçe; tanım aralığını (x), üyelik fonksiyonunu (xmf) ve sayısal giriş verisini (xx) alır ve sayısal gerçek 
giriş veriisinin o üyelik fonksiyonuna olan üyelik değerini hesaplar.
"""

def durulastir(x, LFX, model):
    model = model.lower()
    x = x.ravel()
    LFX = LFX.ravel()
    n = len(x)
    if n != len(LFX):
        print("Bulanık Küme Üyeliği ve Değer Sayısı eşit olmalıdır.")
        return
    # agirlik_merkezi = Ağırlık Merkezi Ortalama
    # maxort = maksimum ortalama
    # minom = en büyüklerin en küçüğü
    # maxom = en büyüklerin en büyüğü
    if 'agirlik_merkezi' in model:
        if 'agirlik_merkezi' in model:
            return agirlik_merkezi(x, LFX)
        elif 'AC0' in model:
            return 0 # AC0(x, mfx)
    elif 'maxort' in model:
        return np.mean(x[LFX == LFX.max()])
    elif 'minom' in model:
        return np.min(x[LFX == LFX.max()])
    elif 'maxom' in model:
        return np.max(x[LFX == LFX.max()])

"""Ağırlık merkezi durulaştırma metodu"""
def agirlik_merkezi(x, LFX):
    sum_moment_area = 0.0
    sum_area = 0.0
    if len(x) == 1:
        return x[0] * LFX[0] / max(LFX[0], np.finfo(float).eps)

    for i in range(1, len(x)):
        x1, x2 = x[i-1], x[i]
        y1, y2 = LFX[i-1], LFX[i]
        # Bu if bloğu FOR'un içinde olmalı:
        if not (y1 == y2 == 0.0 or x1 == x2):
            if y1 == y2:
                moment = 0.5*(x1 + x2)
                area   = y1 * (x2 - x1)
            elif y1 == 0.0 and y2 != 0.0:
                moment = 2/3*(x2-x1) + x1
                area   = 0.5*(x2-x1)*y2
            elif y1 != 0.0 and y2 != 0.0:
                moment = 1/3*(x2-x1) + x1
                area   = 0.5*(x2-x1)*y1
            else:
                moment = (2/3*(x2-x1)*(y2 + 0.5*y1))/(y1+y2) + x1
                area   = 0.5*(x2-x1)*(y1+y2)

            sum_moment_area += moment * area
            sum_area        += area

    return sum_moment_area / max(sum_area, np.finfo(float).eps)


