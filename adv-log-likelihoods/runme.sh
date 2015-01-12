# The expm option is hardcoded into ll.py.

# The last of these three log likelihoods is wrong.

# log likelihood for pade : -1513.00336766
# log likelihood for almohy : -1513.00336766
# log likelihood for eigen : -1516.93448606

FASTA="geneconv-codon/data/YML026C_YDR450W_input.fasta"
STYLE="action"

echo $FASTA
echo $STYLE

#python make-mg-geneconv-json-20150105.py --fasta=$FASTA > geneconv-zero-tau.json
cat geneconv-zero-tau.json | python ll.py > geneconv-zero-tau-$STYLE.txt
cat geneconv-zero-tau-$STYLE.txt | python json-sum.py log_likelihoods
