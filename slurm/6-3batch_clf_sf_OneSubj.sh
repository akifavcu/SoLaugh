for FREQ in delta theta alpha beta gamma; do
  sbatch 6-3clf_sf_OneSubj.sh $FREQ
done