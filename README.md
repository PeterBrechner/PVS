# PVS
PVS.py: Construct a probabilistic volumes of solutions (PVS) following the methodology in Brechner et al. (2026).
PVSgamma.py: Fit particle size distributions to gamma distributions following the methodology in Brechner et al. (2026).
PVSplot.py: Plot representations of a PVS.
PVSplotSix.py: Plot representations of 6 PVSs (3 quantitative regimes for 2 categorical regimes).

Recommended use:
Step 1 - Construct a PVSgamma.fitting() object.
Step 2 - Call get_params() on your fitting() object, which outputs the parameters needed for PVSs to a .MAT file.
Step 3 - Construct a PVSplot.PlotPVS() or PVSplotSix.PlotPVS() object.
Step 4 - Plot representations of your PVS(s) by performing functions on your PlotPVS() object.
