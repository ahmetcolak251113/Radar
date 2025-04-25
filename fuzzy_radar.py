import numpy as np
import fuzzyPy as fuzz
import matplotlib.pyplot as plt

# Giriş değerlerinin tanımlı olduğu aralıklar
x_R = np.arange(0, 91, 1)   # Yol durumu [0-90]
x_W = np.arange(0, 11, 1)   # Hava durumu [0-10]
x_S = np.arange(0, 151, 1)  # Hız [0-150]
x_E = np.arange(0, 21, 1)   # Kullanıcı tecrübesi [0-20]

# Çıkış değerinin tanımlı olduğu aralık
x_O = np.arange(0, 101, 1)  # Çıkış [0-100]

# Üyelik fonksiyonlarını oluşturma
R_kotu    = fuzz.trapez(x_R,   "SOL",   [30, 45])
R_normal  = fuzz.ucgen(x_R, [30, 45, 60])
R_iyi     = fuzz.trapez(x_R,   "SAĞ",   [45, 60])

W_kotu    = fuzz.ucgen(x_W, [0, 0, 5])
W_normal  = fuzz.ucgen(x_W, [0, 5, 10])
W_iyi     = fuzz.ucgen(x_W, [5, 10, 10])

S_az      = fuzz.ucgen(x_S, [0, 0, 70])
S_orta    = fuzz.ucgen(x_S, [0, 70, 130])
S_cok     = fuzz.trapez(x_S,   "SAĞ",   [70, 130])

E_az      = fuzz.ucgen(x_E, [0, 0, 10])
E_orta    = fuzz.ucgen(x_E, [0, 10, 20])
E_cok     = fuzz.ucgen(x_E, [10, 20, 20])

O_az      = fuzz.trapez(x_O,   "SOL",   [25, 50])
O_orta    = fuzz.ucgen(x_O, [25, 50, 85])
O_cok     = fuzz.trapez(x_O,   "SAĞ",   [50, 85])

# Grafik oluşturma
fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, figsize=(6, 10))

# 1) Yol durumu
ax0.plot(x_R, R_kotu,   'r', linewidth=2, label='Kötü')
ax0.plot(x_R, R_normal, 'g', linewidth=2, label='Normal')
ax0.plot(x_R, R_iyi,    'b', linewidth=2, label='İyi')
ax0.set_title("Yol durumu")
ax0.legend()

# 2) Hava durumu
ax1.plot(x_W, W_kotu,   'r', linewidth=2, label='Kötü')
ax1.plot(x_W, W_normal, 'g', linewidth=2, label='Normal')
ax1.plot(x_W, W_iyi,    'b', linewidth=2, label='İyi')
ax1.set_title("Hava durumu")
ax1.legend()

# 3) Sürücü ortalama hızı
ax2.plot(x_S, S_az,   'r', linewidth=2, label='Az')
ax2.plot(x_S, S_orta, 'g', linewidth=2, label='Orta')
ax2.plot(x_S, S_cok,  'b', linewidth=2, label='Çok')
ax2.set_title("Sürücü ortalama hızı")
ax2.legend()

# 4) Kullanıcı tecrübesi
ax3.plot(x_E, E_az,   'r', linewidth=2, label='Az')
ax3.plot(x_E, E_orta, 'g', linewidth=2, label='Orta')
ax3.plot(x_E, E_cok,  'b', linewidth=2, label='Çok')
ax3.set_title("Kullanıcı tecrübesi")
ax3.legend()

# 5) Çıkış: Hız sınırı
ax4.plot(x_O, O_az,   'r', linewidth=2, label='Az')
ax4.plot(x_O, O_orta, 'g', linewidth=2, label='Orta')
ax4.plot(x_O, O_cok,  'b', linewidth=2, label='Çok')
ax4.set_title("Önerilen hız sınırı")
ax4.legend()

plt.tight_layout()
plt.show()

# Inputları alalım
input_R = float(input("Yol Viraj düzeyini girin (0-90): "))
input_W = float(input("Hava durumunu girin (0-10): "))
input_S = float(input("Sürücü ortalama hızını girin (0-150): "))
input_E = float(input("Kullanıcı deneyim yılını girin (0-20): "))

# Üyelik değerlerini hesaplayalım
R_fit_kotu   = fuzz.uyelik(x_R, R_kotu,   input_R)
R_fit_normal = fuzz.uyelik(x_R, R_normal, input_R)
R_fit_iyi    = fuzz.uyelik(x_R, R_iyi,    input_R)

W_fit_kotu   = fuzz.uyelik(x_W, W_kotu,   input_W)
W_fit_normal = fuzz.uyelik(x_W, W_normal, input_W)
W_fit_iyi    = fuzz.uyelik(x_W, W_iyi,    input_W)

S_fit_az     = fuzz.uyelik(x_S, S_az,     input_S)
S_fit_orta   = fuzz.uyelik(x_S, S_orta,   input_S)
S_fit_cok    = fuzz.uyelik(x_S, S_cok,    input_S)

E_fit_az     = fuzz.uyelik(x_E, E_az,     input_E)
E_fit_orta   = fuzz.uyelik(x_E, E_orta,   input_E)
E_fit_cok    = fuzz.uyelik(x_E, E_cok,    input_E)

# Kural tabanı
rule1 = np.fmin(np.fmin(R_fit_kotu,   W_fit_kotu),   O_az)
rule2 = np.fmin(np.fmin(R_fit_normal, W_fit_normal), O_az)
rule3 = np.fmin(np.fmin(R_fit_iyi,    W_fit_iyi),    O_cok)
rule4 = np.fmin(np.fmax(S_fit_az,     E_fit_az),     O_az)
rule5 = np.fmin(np.fmax(S_fit_orta,   E_fit_orta),   O_orta)
rule6 = np.fmin(np.fmax(S_fit_cok,    E_fit_cok),    O_cok)

out_az   = np.fmax(rule1, rule4)
out_orta = np.fmax(rule2, rule5)
out_cok  = np.fmax(rule3, rule6)

# Çıkış grafiği (hata düzeltildi)
O_zeros = np.zeros_like(x_O)
fig, grafik_output = plt.subplots(figsize=(7, 4))

grafik_output.fill_between(x_O, O_zeros, out_az,   color='r', alpha=0.7)
grafik_output.plot(x_O, out_az,   'r--')

grafik_output.fill_between(x_O, O_zeros, out_orta, color='g', alpha=0.7)
grafik_output.plot(x_O, out_orta, 'g--')

grafik_output.fill_between(x_O, O_zeros, out_cok,  color='b', alpha=0.7)
grafik_output.plot(x_O, out_cok,  'b--')

grafik_output.set_title("Periyot çıkışı")
plt.tight_layout()
plt.savefig("cikis.png")

# Durulaştırma
mutlak_bulanik_sonuc = np.fmax(out_az, np.fmax(out_orta, out_cok))
durulastirilmis_sonuc = fuzz.durulastir(x_O, mutlak_bulanik_sonuc, "agirlik_merkezi") * 3/2
print(f'Duru sonuç -> {durulastirilmis_sonuc}')

sonuc_az = fuzz.uyelik(x_O, O_az, durulastirilmis_sonuc)
sonuc_orta = fuzz.uyelik(x_O, O_orta, durulastirilmis_sonuc)
sonuc_cok = fuzz.uyelik(x_O, O_cok, durulastirilmis_sonuc)
print(f'Duru sonuç üyelik değerleri >>> AZ: {sonuc_az}| ORTA: {sonuc_orta} | ÇOK: {sonuc_cok}')

# Varsayılan hız 100 olsun
hizSiniri = 100
hizSiniri -= (sonuc_az * durulastirilmis_sonuc)
hizSiniri += (sonuc_cok * durulastirilmis_sonuc)

if sonuc_az > sonuc_cok:
    hizSiniri = hizSiniri - (sonuc_orta * durulastirilmis_sonuc) /3
degisim = hizSiniri - 100
print(f'Mevcut şartlar altında hız sınırı {degisim} değişerek {hizSiniri} olmalıdır.')
print(f'Değişim oranı %{float(degisim/100)}')



