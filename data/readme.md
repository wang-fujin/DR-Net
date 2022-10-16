The data comes from this paper: Attia, P.M., Grover, A., Jin, N. et al. Closed-loop optimization of fast-charging protocols for batteries with machine learning. Nature 578, 397–402 (2020)

url: https://data.matr.io/1/projects/5d80e633f405260001c0b60a



Since the data points of each cycle are different, to facilitate the model processing, we resampled the data of each cycle to the same number of points. Specifically, 128 points were resampled at equal time intervals from the time,  current, and  voltage  data of the charging process for each cycle, that is, $\mathbf{x}_i\in \mathbb{R}^{128\times 3}$. At the same time, the window length $L$ was set to 8, so $\mathbf{X}_i \in \mathbb{R}^{128\times 24}$.
