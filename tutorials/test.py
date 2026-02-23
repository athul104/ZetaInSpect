import numpy as np
import matplotlib.pyplot as plt
from zetainspect.background import InflationHistory
from zetainspect.scales import Scales
from zetainspect.spectrum import Spectrum
from zetainspect.plotting import plot_quantity
import time

start_time = time.time()
hist = InflationHistory(eta_list=[-0.02, 3.6, -0.6, -0.08],
                        efold_list=[32.0, 30.4, 25.8],
                        r=1e-3)

spec = Spectrum(hist)  # builds Scales internally

N = np.linspace(hist.N_max, hist.N_min, 1000000)

fig, ax = plot_quantity(
    spec,
    y="Pzeta", x="k", mode="ratio",
    N=N,
    ratio=0.005,
    overlays=["pivot", "transitions", "As", "CMB"],
    top_axis_N=True,
    lw=2.0,
)

end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")

plt.show()