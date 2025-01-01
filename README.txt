## This folder contain various main scripts (m) to post-process restart files
#  obtained from MC simulations along with example input files.
#  It also contains a few test codes (t) as well as copies (c)... getting too many
#  so here is the alphabetic list with description of each as reference.

- array_testing.py  :: (t) reading 'restart' file and conversion to spherical coordinates
                as well as binning with histogram and digitize
- ascii_to_binary.py  :: (t) use of module "binascii" for type conversion
- binning_integers.py  :: (t) binning m-components without using histogram
- Cart_to_Sphe.py  :: (t) containing 2 functions to convert from Cartesian to Spherical
               and to bin using histo-digi combination (almost same as used in final)
- Categorical_distrib.py  :: (t) from online library to check scipy functionality
                      for probability of discrete variables (NOT USED at all)
- conv_astype.py  :: (t) short test code for 'astype' function for an array
- convert.py  :: (t) use of 'astype' function m-components read from restart file
- data_exploration.py  :: (t/m in progress) read restart file for different
                   ensembles and print min, max for each component and plot distribution
- extract.py :: (resource) from NRB used to extract mnist data, using it as reference
          for format and some commands
- gaussian_binning.py  :: (t in progress) define Gaussian functions and getting
                   corresponding probabilities for a given angle
- label_from_avg.py  :: (t) reading 'thermal.dat' file and creating labels, now
                 included in the main file
- local_moments.py  :: (m) plotting layer-wise moments by reading restart file
                file name, layer no. and plotting plane as i/p.
- multi_convert.py  :: (t) reads and converts moments to binary file
         'CHECK DIFFERENCE BETWEEN THIS AND convert.py file !!! '
- multi_convert_copy.py  :: (tc) copy of 'multi_convert.py'
         'AGAIN CHECK DIFFERENCE AND MAYBE DELETE ONE? '
- plot_3d_quiver.py  :: (t) plotting 3d-plot with arrows for defined arrays
- random_seq.py  :: (t) using different random number/set generation functions
             available in 'random' library
- Utilities_restart.py  :: (main)
- Utilities_restart.py_copy :: (mc)
- Utilities_restart.py_copy1 :: (mc)
- Utilities_restart.py_diffdim :: (mc)
- Utilities_restart_hist.py  :: (mc)
- Utilities_restart_nolbl.py  :: (mc)
  'All the Utilities* files are different versions of the main file and I need to
   check and decide which to keep and which are safe to delete !!!'