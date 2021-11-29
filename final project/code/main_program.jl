"""
Author: Katherine Kwok
Date: November 27, 2021

This program replicates the code in Arellano (2008). The main steps in the
algorithm are

1. Initialize the parameter values for β, θ, y_hat and an asset grid
   with 200 grid points.

2. Initialize q_0 = 1/(1+r) for all asset levels B' and income levels y.

3. Use value function iteration to solve for the value function and policy
   functions for c, B'. Using the results, construct the optimal income sets
   for repayment A and default D.

4. Update the bond price q_1 so that lenders break-even. Compare q_1 with q_0:
   if the bond price vectors are within a tolerance value, then we have converged.
   If not, update bond prices and repeat value function iteration.

5. Use the data generated above to compute business cycle statistics. If they
   match with the Argentina data, stop, if not, repeat the procedure with adjusted
   parameter values and grid points.
"""


# ---------------------------------------------------------------------------- #
# (1) Benchmark model
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# (2) General model
# ---------------------------------------------------------------------------- #
