import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Create the Tkinter application window
root = tk.Tk()
root.title("Seismogram Generation")
root.geometry("800x600")

# Parameters
R = 100                 # Hypocentral distance in km
v = 3.5                 # velocity in km/sec
ro = 2.7                # density in gm/cc
Mo = 1.2 * 10 ** 22     # seismic moment in dyne cms
fc = 1.07               # corner frequency in Hz
Q = 500                 # Q-Value
fm = 20                 # high cut filter frequency

tt = R / v  # travel time
A0 = (0.85 * Mo) / (4 * np.pi * ro * R * v ** 3)  # calculated source term
f = np.arange(1, 2049) / (4096 * 0.005)
N = len(f)

# Model spectrum
S = (A0 * 4 * np.pi ** 2 * f ** 2) / (1 + f / fc) ** 2
P = np.exp(-np.pi * f * (tt / 224 * f ** 0.93))
H = (1 + (f / fm) ** 8) ** (-0.5)
A = S * P * H
a = 0.0
nf = (np.random.rand(N) - 0.5) * 2
A = A * (1 + a * nf)

# Create a Figure and Axes for the plot
fig1 = plt.figure(figsize=(6, 4), dpi=100)
ax1 = fig1.add_subplot(111)
ax1.loglog(f, A)
ax1.set_xlabel('frequency (Hz)', fontsize=14)
ax1.set_ylabel('log spectral amplitude (cm/sec)', fontsize=14)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# Create a canvas to display the plot
canvas1 = FigureCanvasTkAgg(fig1, master=root)
canvas1.draw()
canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Noise spectrum
t = np.arange(0.005, 10.005, 0.005)
M = len(t)
rd = np.random.rand(M) - 0.5
rd = rd * 2
b = -0.2 * np.log(0.05) / (1 + 0.2 * (np.log(0.2) - 1))
c = b / 0.2
a = (np.exp(1) / 0.2) ** b
Tgm = (1 / fc) + 0.05 * R  # Duration of ground motion
Tn = 2 * Tgm
dt = t[2] - t[1]
w = a * (t / Tn) ** b * np.exp(-c * (t / Tn))
wr = w * rd

# Create a Figure and Axes for the plot
fig2 = plt.figure(figsize=(6, 4), dpi=100)
ax2 = fig2.add_subplot(111)
ax2.plot(t, wr)
ax2.set_xlabel('time (sec)', fontsize=14)
ax2.set_ylabel('windowed white noise', fontsize=14)
ax2.grid(True, linestyle='--', linewidth=0.5)

# Create a canvas to display the plot
canvas2 = FigureCanvasTkAgg(fig2, master=root)
canvas2.draw()

# Fourier Transform of the data
p = 2 ** np.ceil(np.log2(len(wr)))
Nf = np.fft.fft(wr, int(p))
Nf2 = np.abs(Nf * dt)

# Normalization
maxx = np.sqrt(np.mean(Nf2 ** 2))
Nf2 /= maxx

# Create a Figure and Axes for the plot
fig3 = plt.figure(figsize=(6, 4), dpi=100)
ax3 = fig3.add_subplot(111)
ax3.loglog(f, Nf2)
ax3.set_xlabel('frequency (Hz)', fontsize=14)
ax3.set_ylabel('amplitude', fontsize=14)
ax3.grid(True, which='both', linestyle='--', linewidth=0.5)

# Create a canvas to display the plot
canvas3 = FigureCanvasTkAgg(fig3, master=root)
canvas3.draw()

# Multiplication of noise and model spectrum
final = A * Nf2

# Create a Figure and Axes for the plot
fig4 = plt.figure(figsize=(6, 4), dpi=100)
ax4 = fig4.add_subplot(111)
ax4.loglog(f, final)
ax4.set_xlabel('frequency (Hz)', fontsize=14)
ax4.set_ylabel('log spectral amplitude (cm/sec)', fontsize=14)
ax4.grid(True, which='both', linestyle='--', linewidth=0.5)

# Create a canvas to display the plot
canvas4 = FigureCanvasTkAgg(fig4, master=root)
canvas4.draw()

# Generated Seismogram of a region by IFT
phi = np.angle(Nf)
final2 = final * np.exp(1j * phi)
seis = N / 2 * np.fft.ifft(final2) / dt

# Create a Figure and Axes for the plot
fig5 = plt.figure(figsize=(6, 4), dpi=100)
ax5 = fig5.add_subplot(111)
ax5.plot(t, np.real(seis[:2000]))
acc = np.real(seis[:2000])
ax5.set_xlabel('time (sec)', fontsize=14)
ax5.set_ylabel('acceleration (cm/sec^2)', fontsize=14)
ax5.grid(True, linestyle='--', linewidth=0.5)

# Create a canvas to display the plot
canvas5 = FigureCanvasTkAgg(fig5, master=root)
canvas5.draw()

# Generation of observed data
f = f[:409]  # selecting frequency till 20 Hz

dfobs = final2[:409] / (4 * np.pi ** 2 * f ** 2)

# Create a Figure and Axes for the plot
fig6 = plt.figure(figsize=(6, 4), dpi=100)
ax6 = fig6.add_subplot(111)
ax6.loglog(f, np.abs(dfobs), linewidth=2)
ax6.set_xlabel('frequency', fontsize=14)
ax6.set_ylabel('Displacement amplitude', fontsize=14)
ax6.grid(True, which='both', linestyle='--', linewidth=0.5)

# Create a canvas to display the plot
canvas6 = FigureCanvasTkAgg(fig6, master=root)
canvas6.draw()

# 2D grid
L = 101
Dm1 = 100
Dm2 = 20
m1min = 6000
m2min = 100
m1a = m1min + Dm1 * np.arange(0, L)
m2a = m2min + Dm2 * np.arange(0, L)
m1max = m1a[-1]
m2max = m2a[-1]

# Grid search step 1
E = np.zeros((L, L))
for j in range(L):
    for k in range(L):
        dpre = (m1a[j] * 0.85 / (4 * np.pi * ro * v ** 3 * R)) * np.exp(
            -np.pi * tt * f * (1 / m2a[k])
        ) / (1 + (f / fc) ** 2)
        E[j, k] = np.sqrt(np.mean((np.log10(np.abs(dfobs)) - np.log10(np.abs(dpre))) ** 2))

# Find the minimum value of E and the corresponding (a, b) value
rowindex, colindex = np.unravel_index(np.argmin(E), E.shape)
m1est = m1a[rowindex]
m2est = m2a[colindex]

# Grid search step 2
m1a2 = np.arange(m1est - 10, m1est + 11)
L2 = len(m1a2)
E2 = np.zeros(L2)
for j in range(L2):
    dpre2 = (
        m1a2[j]
        * 0.85
        / (4 * np.pi * ro * v ** 3 * R)
        * np.exp(-np.pi * tt * f * (1 / m2est))
        / (1 + (f / fc) ** 2)
    )
    E2[j] = np.sqrt(np.mean((np.log10(np.abs(dfobs)) - np.log10(np.abs(dpre2))) ** 2))

rowindex2 = np.argmin(E2)
m1est2 = m1a2[rowindex2]

# Predicted value from estimated model parameters
dfpre = (m1est2 / R) * np.exp(-np.pi * tt * f * (1 / m2est)) / (1 + (f / fc) ** 2)

# Create a Figure and Axes for the plot
fig7 = plt.figure(figsize=(6, 4), dpi=100)
ax7 = fig7.add_subplot(111)
ax7.loglog(f, np.abs(dfpre), 'r', linewidth=3, label='predicted data')
ax7.loglog(f, np.abs(dfobs), linewidth=3, label='observed data')
ax7.set_xlabel('frequency', fontsize=14)
ax7.set_ylabel('Displacement amplitude', fontsize=14)
ax7.grid(True, which='both', linestyle='--', linewidth=0.5)
ax7.legend()

# Create a canvas to display the plot
canvas7 = FigureCanvasTkAgg(fig7, master=root)
canvas7.draw()

# Create a Figure and Axes for the plot
fig8 = plt.figure(figsize=(6, 4), dpi=100)
ax8 = fig8.add_subplot(111)
ax8.plot(range(1, L2 + 1), E2, 'r', linewidth=2)
ax8.set_xlabel('m1', fontsize=14)
ax8.set_ylabel('root mean square error', fontsize=14)
ax8.grid(True, linestyle='--', linewidth=0.5)

# Create a canvas to display the plot
canvas8 = FigureCanvasTkAgg(fig8, master=root)
canvas8.draw()

# Create buttons to show/hide plots
def show_hide_canvas(canvas, button):
    if canvas.get_tk_widget().winfo_ismapped():
        canvas.get_tk_widget().pack_forget()
        button.config(text= button["text"])
    else:
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        button.config(text=button["text"])

# Create buttons for each plot
button1 = tk.Button(root, text="Model Spectrum", command=lambda: show_hide_canvas(canvas1, button1))
button1.pack()

button2 = tk.Button(root, text="Noise Spectrum", command=lambda: show_hide_canvas(canvas2, button2))
button2.pack()

button3 = tk.Button(root, text="Fourier Transform", command=lambda: show_hide_canvas(canvas3, button3))
button3.pack()

button4 = tk.Button(root, text="Multiplication Result", command=lambda: show_hide_canvas(canvas4, button4))
button4.pack()

button5 = tk.Button(root, text="Seismogram", command=lambda: show_hide_canvas(canvas5, button5))
button5.pack()

button6 = tk.Button(root, text="Generated Seismogram", command=lambda: show_hide_canvas(canvas6, button6))
button6.pack()

button7 = tk.Button(root, text="Grid Search Results", command=lambda: show_hide_canvas(canvas7, button7))
button7.pack()

button8 = tk.Button(root, text="RMS Error", command=lambda: show_hide_canvas(canvas8, button8))
button8.pack()

# Start the Tkinter event loop
root.mainloop()

