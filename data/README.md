# Training data

The FibroinESM reference training set was compiled from published silk gene annotations and UniProt entries:

| file | content |
|---|---|
| train_heavy.fasta | 1,976 heavy-chain fibroin reference proteins |
| train_light.fasta | 45 light-chain fibroin reference proteins |
| train_p25.fasta | 51 P25-type reference proteins |
| train_neg.fasta | 41,216 non-silk proteins |
| test_heavy.fasta | 349 held-out heavy-chain proteins (species-disjoint) |
| test_light.fasta | 8 held-out light-chain proteins |
| test_p25.fasta | 9 held-out P25 proteins |
| test_neg.fasta | 7,273 held-out non-silk proteins |

The test partition is species-disjoint from training so that no tested protein comes from a species present in training.

These FASTA files are not redistributed in this repository because of their size. They are available from the corresponding author upon request.

The 121 insect proteomes screened in this study are listed in Supplementary Table 1 of the associated publication.
