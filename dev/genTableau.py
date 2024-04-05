import argparse
from SDCRK import RKAnalysis, SDCAnalysis


CLI = argparse.ArgumentParser()
CLI.add_argument(
  "--qType",  # name on the CLI - drop the `--` for positional/required parameters
  type=str,
  default='RADAU-RIGHT',  # default if nothing is provided
)
CLI.add_argument(
  "--swList", # LU, BE, FE, TRAP, MIN-SR-S, MIN-SR-NS, BEPAR, ...
  nargs="*",
  type=str,  # any type/callable can be used here
  default=["BE", "LU"]
)
CLI.add_argument(
  "--prSweep", # BE, FE, TRAP, COPY, BEPAR
  type=str,  # any type/callable can be used here
  default="TRAP",
)
CLI.add_argument(
  "--poSweep", #use QUADRATURE or LASTNODE
  type=str,  # any type/callable can be used here
  default="QUADRATURE",
)
CLI.add_argument(
  "--nodes",
  type=int,  # any type/callable can be used here
  default=2,
)
CLI.add_argument(
  "--lowStorage", # for info on lowStorage SDC see: https://doi.org/10.1137/18M117368X
  type=bool,  # any type/callable can be used here, 
  default=False,
)

# parse the command line
args = CLI.parse_args()

# Script parameters
sdcParams = dict(
    # number of nodes : 2, 3, 4, ...
    M=args.nodes,
    # quadrature type : GAUSS, RADAU-RIGHT, RADAU-LEFT, LOBATTO
    quadType=args.qType,
    # node distribution : LEGENDRE, EQUID, CHEBY-1, ...
    distr="LEGENDRE",
    # list of sweeps, ex :
    # -- ["BE"]*3 : 3 sweeps with Backward Euler
    # -- ["BE", "FE"] : BE for first sweep, FE for second sweep
    sweepList=args.swList, # L stable: ["BE", "BEPAR", "OPT-QmQd-0"] with M3 radau right LASTNODE
                                             # not s stable: ["BE", "BEPAR", "OPT-Speck-0"]
                                             # not a stable : ["BEPAR", "BEPAR", "OPT-QmQd-0"]
                                             # A stable: ["BE", "OPT-Speck-0", "OPT-QmQd-0"]
                                             #  not A stable : ["MIN-SR-NS", "OPT-Speck-0", "MIN-SR-NS"]
                                             # L stable? ["MIN-SR-S", "OPT-Speck-0", "MIN-SR-NS"]
                                             # L stable: ["MIN-SR-S", "MIN-SR-S", "MIN-SR-NS"]
                                             # maybe first iteration needs to be BE?
    # initial sweep : COPY (=PIC), or BE, FE, TRAP, ...
    preSweep=args.prSweep,
    # to retrieve the final solution : QUADRATURE or LASTNODE
    postSweep=args.poSweep,
    lowStorage=args.lowStorage,
)

sdc = SDCAnalysis(**sdcParams)
#sdc.printTableau()
