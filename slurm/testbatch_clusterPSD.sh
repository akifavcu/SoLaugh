for FREQ in delta, theta, alpha, beta, gamma; do
  sbatch test_clusterPSD.sh $SUB
done