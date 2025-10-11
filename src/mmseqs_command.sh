HOME_FOLDER=/storage/praha1/home/hamzagamouh

export PATH="$PATH:$HOME_FOLDER/ahoj-allosteric/src/mmseqs/bin"

# mmseqs --help
# mmseqs easy-cluster $HOME_FOLDER/method_1_dataset.fa $HOME_FOLDER/method_1_95_clusterRes tmp --min-seq-id 0.95
mmseqs easy-cluster $HOME_FOLDER/allosteric/method_2/ASD_seqs.fasta $HOME_FOLDER/allosteric/method_2/ASD_30_clusterRes tmp --min-seq-id 0.3