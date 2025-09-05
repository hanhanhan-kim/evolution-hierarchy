import numpy as np
import matplotlib.pyplot as plt
import bokeh.palettes


# Make it look more like Bokeh
plt.style.use('default')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams.update({'font.sans-serif':'Helvetica'})
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.linewidth']= 0.5
plt.rcParams['figure.figsize'] = [6, 6]

# Plotting params:
fig, ax = plt.subplots()
linewidth = 4

# Data and equation params:
m = np.linspace(0.01, 100, 50000)
selecn_by_drifts = [3.0, 1.0, 0.1, 0.0]
colours = bokeh.palettes.Blues256[::50]

# Manually plot the case for m=inf, because np.inf is mishandled otherwise: 
ax.plot(m, len(m)*[1], color=colours[0], linewidth=linewidth, label=r"$\infty$")

# Plot non-infinity 2Ngs values:
non_inf_colours = colours[1::]
for selecn_by_drift, colour in zip(selecn_by_drifts, non_inf_colours):

    num = m * np.exp(selecn_by_drift)
    denom = (1 + m * np.exp(selecn_by_drift))
    
    ax.plot(m, num/denom, color=colour, linewidth=linewidth, label=f"{selecn_by_drift}")

# Stylings:
ax.text(x=4.1, y=0.5, s=r"$P(A) = \frac{me^{2N_es}}{1+me^{2N_es}}$", fontsize=14)
ax.set_xlabel(r"$m = \frac{\mu_{a \rightarrow A}}{\mu_{A \rightarrow a}}$", fontsize=14)
ax.set_ylabel(r"Probability of advantageous allele, $P(A)$", fontsize=14)
ax.set_xscale('log')
ax.legend(loc='best', title=r"$2N_es$")

# plt.savefig("assets/fig2a.svg")
plt.savefig("assets/fig2a.pdf")