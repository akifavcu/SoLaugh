for FREQ in delta theta alpha beta gamma; do
  sbatch 6-1clf-SingleFeatures.sh $FREQ
done