This is like adv-log-likelihoods except for the following changes
to the input and output interface.

input changes:
 * input of shape (nsites, ) specifying a weight for each site
 * ordered list of edge indices for which to compute edge length derivatives

output changes:
 * the weighted sum of site log likelihoods instead of the vector
 * an array of edge length derivatives

NOTE:
moved to https://github.com/argriffing/jsonctmctree
